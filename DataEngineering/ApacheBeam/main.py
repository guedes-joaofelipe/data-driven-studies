import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io.textio import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions

pipeline_options = PipelineOptions()
pipeline = beam.Pipeline(options=pipeline_options)

def text_to_list(element, delimiter='|'):
    return element.split(delimiter)
    
columns = [
    'id',
    'data_iniSE',
    'casos', 
    'ibge_code', 
    'cidade', 
    'uf', 
    'cep', 
    'latitude',
    'longitude'
]

def list_to_dict(element, columns):
    return dict(zip(columns, element))

def treat_date(element):
    element['year_month'] = '-'.join(element['data_iniSE'].split('-')[:2])

    return element

def key_uf(element):
    key = element['uf']
    return (key, element)

def dengue_cases(element):
    uf, registers = element
    for register in registers:
        yield (f'{uf}-{register["year_month"]}', float(register['casos']))

def key_uf_year_month(element):
    data, mm, uf = element
    year_month = '-'.join(data.split('-')[:2])
    mm = 0.0 if float(mm) < 0 else float(mm)        
    return f'{uf}-{year_month}', mm

def arredonda(element):
    chave, mm = element
    return (chave, round(mm, 1))

def filter_empty_fields(element):
    chave, data = element
    if all([data['chuvas'], data['dengue']]):
        return True

    return False

def unzip_elements(element):    
    key, data = element

    state, year, month = key.split('-')
    rain = data['chuvas'][0]
    dengue = data['dengue'][0]

    return (state, year, month, rain, dengue)

def prepare_csv(element, delimiter=';'):
    return f"{delimiter}".join([str(e) for e in element])

# dengue is a PCollection
dengue = (
    pipeline
    | "Leitura do dataset de dengue" >> 
        ReadFromText('./data/sample_casos_dengue.txt', skip_header_lines=1)
    | "From text to list" >> beam.Map(text_to_list)
    | "From list to dict" >> beam.Map(list_to_dict, columns)
    | "Add year_month" >> beam.Map(treat_date)
    | "Create key by state" >> beam.Map(key_uf)
    | "GroupBy State" >> beam.GroupByKey()
    | "Unzip dengue cases" >> beam.FlatMap(dengue_cases) # Permite a utilização de yield
    | "Sum of cases by key" >> beam.CombinePerKey(sum)
    # | "Show results" >> beam.Map(print)
)

chuvas = (
    pipeline
    | "Read rain dataset" >> ReadFromText('./data/sample_chuvas.csv', skip_header_lines=1)
    | "From text to list (rain)" >> beam.Map(text_to_list, delimiter=',')
    | "Create key UF-YEAR-MONTH" >> beam.Map(key_uf_year_month)    
    | "Sum of total rain by key" >> beam.CombinePerKey(sum) # operacao pesada
    | "Round rain results" >> beam.Map(arredonda)
    # | "Show results" >> beam.Map(print)
)

result = (
    # (chuvas, dengue)
    # | "Pile PCollections" >> beam.Flatten()
    # | "GroupByKey" >> beam.GroupByKey()
    ({'chuvas': chuvas, 'dengue': dengue})
    | "Merge PCollections" >> beam.CoGroupByKey()
    | "Filter empty data" >> beam.Filter(filter_empty_fields)
    | "Unzip elements" >> beam.Map(unzip_elements)
    | "Prepare csv" >> beam.Map(prepare_csv)
    # | "Show union results" >> beam.Map(print)
)

header = 'state;year;month;rain;dengue'
result | 'Create CSV file' >> WriteToText(
    './data/result', 
    file_name_suffix='.csv', 
    num_shards= 2,# number of files used for output
    header=header
)

pipeline.run()
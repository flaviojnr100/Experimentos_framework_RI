8 ExperimentosframeworkRI

# Frameworks: 
cherche == 1.0.1;
farm-haystack == 1.20.0rc0;
pyserini == 0.21.0;
python-terrier == 0.9.2

# Elasticsearch == 8.8.2
Utilizado nas implementações dos frameworks cherche e haystack.
Antes de fazer qualquer experimento que utilize essa ferramenta, é necessario fazer a indexação na ferramenta. Salvar a coleção de documentos no formato JSON seguindo o padrão a seguir: {"content":"documento","name":"Titulo do documento"}.
Cada configuração dos exeperimentos, foi criado índices especificos para ada configuração e seguindo o mesmo padrão do arquivo JSON.

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pickle
import pandas as pd

app = Flask(__name__)
api = Api(app)

# Carregando o modelo e as features
model = pickle.load(open('model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# Mapeamento de nomes de colunas sem espaço para nomes com espaço ou caracteres especiais
column_mapping = {
    "work_industry_Financial_Services": "work_industry_Financial Services",
    "work_industry_Health_Care": "work_industry_Health Care",
    "work_industry_Investment_Banking": "work_industry_Investment Banking",
    "work_industry_Investment_Management": "work_industry_Investment Management",
    "work_industry_Media_Entertainment": "work_industry_Media/Entertainment",
    "work_industry_Nonprofit_Gov": "work_industry_Nonprofit/Gov",
    "work_industry_PE_VC": "work_industry_PE/VC",
    "work_industry_Real_Estate": "work_industry_Real Estate"
}


class Predict(Resource):
    def post(self):
        # Recebe os dados JSON da requisição
        data = request.get_json()

        # Converte os dados para um DataFrame
        input_data = pd.DataFrame([data])

        # Converter valores booleanos
        if 'international' in input_data.columns:
            input_data['international'] = input_data['international'].apply(lambda x: 1 if x else 0)

        # Aplicar mapeamento de colunas
        input_data.rename(columns=column_mapping, inplace=True)

        # Verifica se todas as colunas esperadas pelo modelo estão presentes
        for feature in features:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Adiciona colunas ausentes como 0

        # Ordena as colunas no mesmo formato das features
        input_data = input_data[features]

        # Faz a previsão usando o modelo
        prediction = model.predict(input_data)

        return jsonify({'prediction': int(prediction[0])})  # Envia o resultado da previsão


# Adiciona a rota /predict para a API
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(port=5002)

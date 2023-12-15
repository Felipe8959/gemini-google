import google.generativeai as genai
import PIL.Image

genai.configure(api_key="AIzaSyDt8a19qwbZ_bJrweUJqwIriK61cwcFWgY")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel('gemini-pro-vision')

img = PIL.Image.open('pt_131.png')

#response = model.generate_content(img)

#print("Resposta 1:", response.text)

response = model.generate_content(["Este é um trecho de uma Norma, informe as solicitações da mesma de uma forma que qualquer um compreenda", img])
response.resolve()

print(response.text)
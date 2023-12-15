import google.generativeai as genai

genai.configure(api_key="AIzaSyDt8a19qwbZ_bJrweUJqwIriK61cwcFWgY")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Me dê idéias de como utilizar o Gemini para ajudar uma empresa que realiza testes conforme Normas NBR e IEC", stream=True)

for chunk in response:
    print(chunk.text)

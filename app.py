from flask import Flask,request,render_template
from flask_cors import CORS, cross_origin
import pickle


app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def Home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            fnlwgt = float(request.form['fnlwgt'])
            education_num=int(request.form['education_num'])
            capital_gain = int(request.form['capital_gain'])
            capital_loss = int(request.form['capital_loss'])
            hours_per_week = int(request.form['hours_per_week'])
            workclass_Local_gov = int(request.form['workclass_ Local-gov'])
            workclass_Never_worked = int(request.form['workclass_ Never-worked'])
            workclass_Private = int(request.form['workclass_ Private'])
            workclass_Self_emp_inc = int(request.form['workclass_ Self-emp-inc'])
            workclass_Self_emp_Not_inc = int(request.form['workclass_ Self-emp-Not-inc'])
            workclass_State_gov = int(request.form['workclass_ State-gov'])
            workclass_0 = int(request.form['workclass_0'])
            education_Assoc_voc = int(request.form['education_ Assoc-voc'])
            education_Bachelors = int(request.form['education_ Bachelors'])
            education_Doctorate = int(request.form['education_ Doctorate'])
            education_HS_grad = int(request.form['education_ HS-grad'])
            education_Masters = int(request.form['education_ Masters'])
            education_Preschool = int(request.form['eucation_ Preschool'])
            education_Prof_school = int(request.form['eucation_ Prof-school'])
            education_Some_college = int(request.form['education_ Some-college'])
            education_Primary = int(request.form['education_Primary'])
            marital_status_Married_civ_spouse = int(request.form['marital_status_ Married-civ-spouse'])
            marital_status_Married_spouse_absent = int(request.form['marital_status_ Married-spouse-absent'])
            marital_status_Never_married = int(request.form['marital_status_ Never-married'])
            marital_status_Separated = int(request.form['marital_status_ Separated'])
            marital_status_Widowed = int(request.form['marital_status_ Widowed'])
            occupation_Craft_repair = int(request.form['occupation_ Craft-repair'])
            occupation_Exec_managerial = int(request.form['occupation_ Exec-managerial'])
            occupation_Farming_fishing = int(request.form['occupation_ Farming-fishing'])
            occupation_Handlers_cleaners = int(request.form['occupation_ Handlers_cleaners'])
            occupation_Machine_up_inspct = int(request.form['occupation_ Machine-up-inspct'])
            occupation_Other_service = int(request.form['occupation_ Other-service'])
            occupation_Priv_house_serv = int(request.form['occupation_ Priv-house-serv'])
            occupation_Prof_specialty = int(request.form['occupation_ Prof-specialty'])
            occupation_Protective_serv = int(request.form['occupation_ Protective-serv'])
            occupation_Sales = int(request.form['occupation_ Sales'])
            occupation_Tech_support = int(request.form['occupation_ Tech-support'])
            occupation_Transport_moving = int(request.form['occupation_ Transport-moving'])
            occupation_0 = int(request.form['occupation_0'])
            relationship_Not_In_family = int(request.form['relationship_ Not-In-family'])
            relationship_Other_relative = int(request.form['relationship_ Other-relative'])
            relationship_Own_child = int(request.form['relationship_ Own-child'])
            relationship_Unmarried = int(request.form['relationship_ Unmarried'])
            relationship_Wife = int(request.form['relationship_ Wife'])
            race_Asian_Pac_Islander = int(request.form['race_ Asian-Pac-Islander'])
            race_Black = int(request.form['race_ Black'])
            race_Other = int(request.form['race_ Other'])
            race_White = int(request.form['race_ White'])
            sex_Male = int(request.form['sex_ Male'])
            native_country_Central_America = int(request.form['native_country_Central_America'])
            native_country_EU = int(request.form['native_country_EU'])
            native_country_North_America = int(request.form['native_country_North_America'])
            native_country_South_America = int(request.form['native_country_South_America'])

            filename = "XGB_model.pickle"
            loaded_model = pickle.load(open(filename, 'rb'))
            prediction = loaded_model.predict([[age,fnlwgt,education_num,capital_gain,capital_loss,hours_per_week,workclass_Local_gov,workclass_Never_worked,workclass_Private,workclass_Self_emp_inc,workclass_Self_emp_Not_inc,
                                                workclass_State_gov,workclass_0,education_Assoc_voc,education_Bachelors,education_Doctorate,education_HS_grad,education_Masters,education_Preschool,education_Prof_school, education_Some_college, education_Primary,marital_status_Married_civ_spouse,
                                                marital_status_Married_spouse_absent,marital_status_Never_married,marital_status_Separated, marital_status_Widowed,
                                                occupation_Craft_repair,occupation_Exec_managerial,occupation_Farming_fishing, occupation_Handlers_cleaners,occupation_Machine_up_inspct,occupation_Other_service,occupation_Priv_house_serv,occupation_Prof_specialty,
                                                occupation_Protective_serv,occupation_Sales,occupation_Tech_support,occupation_Transport_moving,occupation_0,
                                                relationship_Not_In_family,relationship_Other_relative,relationship_Own_child,relationship_Unmarried,relationship_Wife,
                                                race_Asian_Pac_Islander,race_Black,race_Other,race_White,sex_Male,native_country_Central_America,native_country_EU,native_country_North_America,native_country_South_America]])
            return render_template('results.html', prediction=prediction[0])
        except Exception as e:
            print("The Exception message is:",e)
            return "Something is wrong"
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port='8080')
    #app.run(debug=True)
import numpy as np
import pickle
import streamlit as st


# Load the saved model
loaded_model = pickle.load(open('trained_model1.sav', 'rb'))


# Function for making predictions
def student_dropout_prediction(input_data):
    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make the prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Interpret the prediction result
    if prediction[0] == 0:
        return 'The student is Dropout'
    elif prediction[0] == 1:
        return 'The student is Graduate'
    else:
        return 'The student is Enrolled'


def main():
    # Streamlit app title
    st.title('Student Dropout Prediction Web App')
    
    # User input for each feature
    Application_mode = st.text_input('Application Mode (1-18)')
    Mothers_qualification = st.text_input("Mother's Qualification (1-29)")
    Mothers_occuption = st.text_input("Mother's Occupation (1-32)")    
    Fatherss_occuption = st.text_input("Father's Occupation (1-46)")
    Debtor = st.text_input('Debtor (0-3)')
    Tuition_fees_up_to_date = st.text_input('Tuition Fees Up-to-date (0-2.5)')
    Gender = st.text_input('Gender (0-5)')
    Scholarship_holder = st.text_input('Scholarship Holder (0-1.5)')
    Age_at_enrollment = st.text_input('Age at Enrollment (17-60)')
    Curricularunits_1st_sem_credited = st.text_input('1st Sem Units Credited (0-10)')
    Curricularunits_1st_sem_enrolled = st.text_input('1st Sem Units Enrolled (0-15)')
    Curricularunits_1st_sem_evaluation = st.text_input('1st Sem Units Evaluation (0-35)')
    Curricularunits_1st_sem_approved = st.text_input('1st Sem Units Approved (0-16)')
    Curricularunits_1st_sem_grade = st.text_input('1st Sem Grade (0-20)')
    Curricularunits_2nd_sem_credited = st.text_input('2nd Sem Units Credited (0-11)')
    Curricularunits_2nd_sem_enrolled = st.text_input('2nd Sem Units Enrolled (0-16.5)')
    Curricularunits_2nd_sem_evaluation = st.text_input('2nd Sem Units Evaluation (0-36)')
    Curricularunits_2nd_sem_approved = st.text_input('2nd Sem Units Approved (0-17)')
    Curricularunits_2nd_sem_grade = st.text_input('2nd Sem Grade (0-21)')

    # Initialize an empty diagnosis string
    diagnosis = ''

    # Button for making predictions
    if st.button('Get Dropout Prediction'):
        # Collect all input values into a list for prediction
        input_data = [Application_mode, Mothers_qualification, Mothers_occuption, Fatherss_occuption, Debtor,
                      Tuition_fees_up_to_date, Gender, Scholarship_holder, Age_at_enrollment, 
                      Curricularunits_1st_sem_credited, Curricularunits_1st_sem_enrolled, Curricularunits_1st_sem_evaluation,
                      Curricularunits_1st_sem_approved, Curricularunits_1st_sem_grade, Curricularunits_2nd_sem_credited,
                      Curricularunits_2nd_sem_enrolled, Curricularunits_2nd_sem_evaluation, Curricularunits_2nd_sem_approved,
                      Curricularunits_2nd_sem_grade]
        
        # Perform the prediction
        diagnosis = student_dropout_prediction(input_data)
        
    # Display the result
    st.success(diagnosis)


if __name__ == '__main__':
    main()

import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
#import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import json
import openai
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
import warnings

#openai_key = os.getenv("OPENAI_KEY")
openai_key = st.secrets("OPENAI_API_KEY")

def main():

    load_dotenv()

    model_4 = 'gpt-4-1106-preview'
    model_3 = 'gpt-3.5-turbo-1106'

    def csv_agent_func(file_path, user_question):
        """ Run the CSV agent witht the given file path and user message."""
        llm = ChatOpenAI(temperature=0, model=model_3)
        agent = create_csv_agent(llm, file_path, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)

        try:
            #Format the user's input and wrap it with the required input key
            tool_input = {
                "input": {
                    "name": "python",
                    "arguments": user_question
                }
            }

            response = agent.run(tool_input)
            return response
        except Exception as e:
            st.write(f"Error: {e}")
            return None
        
    def display_content_from_json(json_response):
        """Display the content to streamlit based on the structure of the json response."""
        #check if the response has a plain text
        if "answer" in json_response:
            st.write(json_response["answer"])
        #check if the response has a bar chart
        if "bar" in json_response:
            data = json_response["bar"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)
        
        # check if the response has a table
        if "table" in json_response:
            data = json_response["table"]
            df = pd.DataFrame(data)
            df.set_index(data["data"], columns=data["columns"])
            st.table(df)
    
    def extract_code_from_response(response):
        """Extract the python code from a string response"""
        #use a regex pattern to match content between triple backticks
        code_pattern  = r"```pyton(.*?)```"
        match = re.search(code_pattern, response, re.DOTALL)

        if match:
            #extract the matched code and strip and leading/trailing spaces
            return match.group(1).strip()
        return None
    
    """Main streamlit application for csv analysis"""
    st.title("Chat with your CSV 📊")
    st.write("Please upload a CSV file to get started")

    uploaded_file  = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)

        #save the file to disk
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)

        user_input = st.text_input("Ask a question about the CSV file")
        if st.button("Ask"):
            #run the csv agent
            response = csv_agent_func(file_path, user_input)

            #extracting code from the response
            code_to_execute = extract_code_from_response(response)

            if code_to_execute:
                #execute the code
                try:
                    #display_content_from_json(json.loads(response))
                    #making df available for execution in the context
                    exec(code_to_execute, globals(), {"df": df})
                    #fig = plt.show() #get the current figure
                    st.pyplot() #display the figure using streamlit
                except Exception as e:
                    st.write(f"Error executing code: {e}")
            else:
                st.write(response)



if __name__ == '__main__':
    main()
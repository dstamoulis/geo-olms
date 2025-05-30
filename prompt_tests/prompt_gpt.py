from openai import OpenAI
client = OpenAI()
import prompt

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": 
            f'''
            {prompt.INIT_WORKFLOW_PROMPT}\n
            {prompt.TASK_EXECUTION_PROMPT}\n

            For example:
            {prompt.INIT_WORKFLOW_TEMPLATE}\n

            Provide the output in the following format:
            {prompt.INIT_TEMPLATE}\n

            Here is the task to be executed:

            Fetch xView1 images from Austin International Airport. Consider a wide area. Then run the Swin-L detector and finally please zoom the map there!
            '''
       
        }
    ]
)

print(completion.choices[0].message.content)
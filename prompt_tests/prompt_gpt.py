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

            Provide the output in the same format as the example above. Make sure you choose only the agents provided in the example to complete the task.

            Here is the task to be executed:

            'Fetch BigEarthNet images from June 2017. Then run the ResNet-32 LCC classifer on the images. Last, plot the LCC classes of category 'Non-irrigated arable land'.'
            '''
       
        }
    ]
)

print(completion.choices[0].message.content)
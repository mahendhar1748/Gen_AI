import boto3
import json

prompt_data = '''
Act as Shakespeare and write a poem on Nature 
'''

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt":prompt_data,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9 
}

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"
response = bedrock.invoke_model(
    body= body,
    modelId=model_id,
    accept = "application/json",
    contentType="application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation']
print(response_text)

#============================++++++++++++ OUTPUT++++++++===========================================
"""
Nature, a poem by William Shakespeare 

When in the chronicle of wasted time
I see descriptions of the fairest wights,
And beauty making beautiful old rhyme,
In praise of ladies dead and lovely knights,

Then, in the blazon of sweet beauty's best,
Of hand, of foot, of lip, of eye, of brow,
I see their antique pen would have express'd
Their beauty's pattern to eternity:

But since, neglected, all unwatch'd,
Die, and their beauty is no more in sight,
Nor in their golden statues shall they be
In earthly pomp, in tomb, in monument,

Yet shall their beauty live in poetry,
And in the chronicle of time, be told
Their virtues, and their praise, in verse, shall be
In spite of death, their beauty shall not fade.
"""
import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4o')

print(encoder.n_vocab)

text = "Hello world"

tokens = encoder.encode(text)

output =  encoder.decode(tokens)
print(tokens,output)

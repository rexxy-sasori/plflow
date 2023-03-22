import transformers

module = transformers

"""
Import all huggingface API that contains "AutoModel" 
"""
for attribute_name in dir(module):
    if 'AutoModel' in attribute_name:
        attribute = getattr(module, attribute_name)
        locals()[attribute_name] = attribute

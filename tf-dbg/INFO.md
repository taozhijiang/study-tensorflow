```bash
~ saved_model_cli show --dir saved_simple_add/00001 --all
MetaGraphDef with tag-set: 's, e, r, v, e' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x-input1'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 1)
        name: x-input1:0
    inputs['x-input2'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 1)
        name: x-input2:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y-output'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 1)
        name: y-output:0
  Method name is: tensorflow/serving/predict
~
~
```
# ImageDoctored

### generate

Directory structure:

area: xxx.jpg ...

class:

&emsp; a:
      
&emsp; &emsp; a000.jpg
    
&emsp; &emsp; ...
    
&emsp; b:

&emsp; ...

bounding_detect: xxx.jpg ...

configs: xxx.json ...

There are three types of manipulation: slicing, copy-move, removal. Run generate_configs() first then run them.



### attack

```shell
python generator.py
mkdir pretrained
```

uncomment patch extractor in 'train.py'

```shell
python train.py
```

preparing testing files and test

```shell
mkdir testing
cd testing
mkdir authentic # put authentic images here
mkdir doctored # put doctored images here
python eval.py
```

attack

```python
python attack.py
```


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

```shell
mkdir images
python attack.py
```



### attack ELA

```shell
mkdir patches
cd patches
mkdir authentic
mkdir doctored
mkdir attack
```

run `generate_patches` in 'patch_generator.py' then run 'train_ela.py' then run `repaint` in 'patch_generator.py'




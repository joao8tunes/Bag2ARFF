## Bag2Arff
This script allows to convert a typical Doc-Attribute text representation matrix to an Arff file based in Weka API format. Basically, this script can convert any Doc-Attribute text format into a sparse Arff file dispensing the use of Weka API. This script also allow the tokenization of composite classes like "category-polarity" or "topic-semantic", characterizing the two levels of semantic complexity.
> Converting a collection of documents:
```
python3 Bag2Arff.py --token - --input out/Bag/txt/ --output out/Bag/arff/
```


### Related scripts
* [Bag2Arff.py](https://github.com/joao8tunes/Bag2Arff/blob/master/Bag2Arff.py)


### Assumptions
These scripts expect a database folder following an specific hierarchy like shown below:
```
in/db/                 (main directory)
---> class_1/          (class_1's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> class_2/          (class_2's directory)
---------> file_1      (text file)
---------> file_2      (text file)
---------> ...
---> ...
```


### Observations
All generated files use *TAB* character as a separator.


### Requirements installation (Linux)
> Python 3 + PIP installation as super user:
```
apt-get install python3 python3-pip
```
> Scipy installation as normal user:
```
pip3 install -U scipy
```


### See more
Project page on LABIC website: http://sites.labic.icmc.usp.br/MSc-Thesis_Antunes_2018

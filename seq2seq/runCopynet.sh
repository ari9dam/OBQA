
rm -rf copynet/output
mkdir copynet/output
allennlp train copynet/copynet_novocab.json -s copynet/output/ --include-package 



rm -rf Seq2Seq_General/output
mkdir Seq2Seq_General/output
allennlp train Seq2Seq_General/seq2seq.json -s Seq2Seq_General/output/


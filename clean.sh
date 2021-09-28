rm -r model_record/text_model_weights

for i in $(seq 1 30)
do
    rm -r model_record/text_model_weight_$i
done

cp -r model_record/test_bed model_record/text_model_weights
rm -r log.txt

for i in $(seq 1 30)
do
    python3 sst_complete.py | tee -a log.txt
    cp -r model_record/text_model_weights model_record/text_model_weight_$i
done

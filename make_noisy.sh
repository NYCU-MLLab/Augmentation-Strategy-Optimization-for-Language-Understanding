# stack eda
textattack augment --input-csv $1_test.csv --output-csv $1_stack_eda.csv --input-column sentence --recipe eda --pct-words-to-swap .1 --transformations-per-example 5 --exclude-original --overwrite

# eda
textattack augment --input-csv $1_test.csv --output-csv $1_eda.csv --input-column sentence --recipe eda --pct-words-to-swap .1 --transformations-per-example 4 --exclude-original --overwrite

# word embedding
textattack augment --input-csv $1_test.csv --output-csv $1_embedding.csv --input-column sentence --recipe embedding --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite

# clare
textattack augment --input-csv $1_test.csv --output-csv $1_clare.csv --input-column sentence --recipe clare --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite

# checklist
textattack augment --input-csv $1_test.csv --output-csv $1_checklist.csv --input-column sentence --recipe checklist --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite

# charswap
textattack augment --input-csv $1_test.csv --output-csv $1_char.csv --input-column sentence --recipe charswap --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite

# backtranslation de
python3 create_noisy.py --input-csv $1_test.csv --output-csv $1_backtrans_de.csv --recipe backtrans --language de --input-column sentence --overwrite --transformations-per-example 1

# backtranslation ru
python3 create_noisy.py --input-csv $1_test.csv --output-csv $1_backtrans_ru.csv --recipe backtrans --language ru --input-column sentence --overwrite --transformations-per-example 1

# backtranslation zh
python3 create_noisy.py --input-csv $1_test.csv --output-csv $1_backtrans_zh.csv --recipe backtrans --language zh --input-column sentence --overwrite --transformations-per-example 1

# spelling
python3 create_noisy.py --input-csv $1_test.csv --output-csv $1_spell.csv --input-column sentence --recipe spell --overwrite --transformations-per-example 3

# textfooler - failed
# python3 attack.py --datapath-prefix $1 --attack-method textfooler --target-model roberta-base

# fast - failed
# python3 attack.py --datapath-prefix $1 --attack-method fast-alzantot --target-model roberta-base

# iga - failed
# python3 attack.py --datapath-prefix $1 --attack-method iga --target-model roberta-base

# BAE - failed
# python3 attack.py --datapath-prefix $1 --attack-method bae --target-model roberta-base

# deepwordbug - ok
python3 attack.py --datapath-prefix $1 --attack-method deepwordbug --target-model roberta-base

# pwws - ok
python3 attack.py --datapath-prefix $1 --attack-method pwws --target-model roberta-base

# input-reduction - ok
python3 attack.py --datapath-prefix $1 --attack-method input-reduction --target-model roberta-base

# kuleshov - failed - two days
# python3 attack.py --datapath-prefix $1 --attack-method kuleshov --target-model roberta-base

# pso - failed
# python3 attack.py --datapath-prefix $1 --attack-method pso --target-model roberta-base

# textbugger - ok
python3 attack.py --datapath-prefix $1 --attack-method textbugger --target-model roberta-base

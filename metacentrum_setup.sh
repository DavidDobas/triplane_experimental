export TMPDIR=$SCRATCHDIR
trap 'clean_scratch' TERM EXIT
module add python/python-3.10.4-intel-19.0.4-sc7snnf
python3 -m venv $TMPDIR/venv
$TMPDIR/venv/bin/pip install --upgrade pip
$TMPDIR/venv/bin/pip install -r requirements.txt
source $TMPDIR/venv/bin/activate
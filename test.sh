for run in {1..100}; do
  echo $run
  python3 Train.py &> /dev/null
done

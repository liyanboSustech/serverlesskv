for pid in $(ps -ef | grep 'ipykernel_launcher' | grep -v grep | awk '{print $2}')
do
    echo "Terminating process $pid"
    kill -9 $pid
done
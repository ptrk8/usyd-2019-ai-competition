
declare -a FILE_NAMES=(task1.py task2.py task3.py)


for script in "${FILE_NAMES[@]}"
do
    echo ${script}
    # Forces script to be true https://stackoverflow.com/questions/11231937/bash-ignoring-error-for-a-particular-command
    python ${script}
done
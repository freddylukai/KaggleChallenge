Steps to create tf records:

1. Clone the repo in the VM. And inside the VM execute the following steps.
2. go to repo_path/src/xray-model/
3. execute: chmod +x create_records_and_copy_parallel_aug.sh
4. execute nohup ./create_records_and_copy_parallel_aug.sh &

The output will be written to nohup.out, you can safely close your connection and the process will run in the background.

To check how the process is doing:

ps -aux | grep create_records_and_copy_parallel_aug.sh

NOTE: This assumes you have installed python 3.* as 'python3'. You may need to modify the sh files in the VM
if is installed as "python".

NOTE2: if you want to generate the tf records without augmentation do all the above and replace
create_records_and_copy_parallel_aug.sh by create_records_and_copy.sh
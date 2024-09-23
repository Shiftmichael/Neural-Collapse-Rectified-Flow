

score =1 
file_name = "results/reflow/score_record.txt"
with open(file_name, 'a') as file:
    record_id = 0
    file.write(f"Record {record_id}: FID: {score}\n")
    record_id += 1
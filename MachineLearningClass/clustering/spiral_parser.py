import csv
def spiral_data_parser():
    file='spiral.txt'
    output_file = "output_spiral.csv"
    with open(file, 'r') as fl, open(output_file, "w", newline="") as csvfile:
        reader = csv.reader(fl, delimiter='\t')
        writer = csv.writer(csvfile)
        for row in reader:
            writer.writerow([float(val) for val in row])
    return output_file
import csv

def load_csv(path:str):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|')
        result = []
        for val in csv_reader:
            result.append(val)
        return result

"""
clean_customer: Cleans customers so that customers to orders shares a 1:N relation
"""
def clean_customers(order_path:str, customers_path:str, output_file:str):

    order_set = set()
    with open(order_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
                
            for val in csv_reader:
                order_set.add(val[1])

    with open(output_file, 'w', newline="") as f:
        writer = csv.writer(f, delimiter='|')

        with open(customers_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')

                for val in csv_reader:
                    if val[0] in order_set:
                        writer.writerow(val)








import pandas as pd

# I have a .csv file with the following columns: img id, width, height,Pixels,Nr of blocks,Mem Bitstream, Parsing time,First column DPCM to YCCC, YCCC to BayerGB, Mem transfer device host, Total time. Import the file statistics_blocks.csv. 

pd.set_option('display.width', 1000)  # Set the display width to 1000 characters
pd.set_option('display.max_columns', None) 

# Read the CSV file
df = pd.read_csv('C:/DATA/Repos/OMLS_Masters_SW/images/wallpapers/statistics_blocks.csv')

# add the column names
df.columns = ['img id', 'width', 'height', 'Pixels', 'Nr of blocks', 'Mem Bitstream', 'Parsing time', 'First column', 'DPCM to YCCC', 'YCCC to BayerGB', 'Mem device 2 host', 'Total time']

# Access the data in the DataFrame
# Example: print the first 5 rows

# sort the dataframe by column img id
df = df.sort_values(by=['img id', 'Nr of blocks'])

print(df.head(50))
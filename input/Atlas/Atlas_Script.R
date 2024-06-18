library(httr)
library(jsonlite)


# Define the API URL
url <- "https://atlas.openforestprotocol.org/_next/data/xZ8P0BbE31c_v-wWYLjCL/index.json"  

# Make the GET request
response <- GET(url)

# Check the status of the response
if (status_code(response) == 200) {
  # Parse the response content
  content <- content(response, as = "text", encoding = "UTF-8")
  
  # Convert the JSON content to a list
  data_list <- fromJSON(content, flatten = TRUE)
  
  # Convert the list to a data frame
  data_frame <- as.data.frame(data_list)
  
  # Display the data frame
  print(data_frame)
} else {
  print(paste("Request failed with status code:", status_code(response)))
}






#Removing the prefix from columns 6 to 26

# Print original column names
print("Original column names:")
print(colnames(data_frame))

# Identify the columns that need modification (from the 6th to the last column)
cols_to_modify <- 6:ncol(data_frame)

# Remove the specified text from the column names for the identified columns
colnames(data_frame)[cols_to_modify] <- gsub("^pageProps\\.projects\\.hits\\.hits\\._source\\.", "", colnames(data_frame)[cols_to_modify])

# Print modified column names
print("Modified column names:")
print(colnames(data_frame))

# Display the data frame with modified column names
print(data_frame)



# Removing the prefix for the first 5 columns

# Print original column names
print("Original column names:")
print(colnames(data_frame))

# Identify the first 5 columns and the remaining columns
cols_first_five <- 1:5
cols_remaining <- 6:ncol(data_frame)

# Remove the prefix from the first 5 column names
colnames(data_frame)[cols_first_five] <- gsub("^pageProps\\.projects\\.hits\\.hits\\._", "", colnames(data_frame)[cols_first_five])

# Remove the prefix from the remaining column names
colnames(data_frame)[cols_remaining] <- gsub("^pageProps\\.projects\\.hits\\.hits\\._source\\.", "", colnames(data_frame)[cols_remaining])

# Print modified column names
print("Modified column names:")
print(colnames(data_frame))

# Display the data frame with modified column names
print(data_frame)













# Function to convert list columns to character
convert_list_columns <- function(df) {
  for (col in colnames(df)) {
    if (is.list(df[[col]])) {
      df[[col]] <- sapply(df[[col]], toString)
    }
  }
  return(df)
}

# Apply the function to your data frame
data_frame <- convert_list_columns(data_frame)

# Print modified column names
print("Modified column names:")
print(colnames(data_frame))

# Display the data frame with modified column names
print(data_frame)




# Write the data frame to a CSV file
write.csv2(data_frame, "Atlas_data_frame.csv")

# Write the data frame to a JSON file
write_json(data_frame, "Atlas_data_frame.json", pretty = TRUE)

# Confirmation message
print("Data frame has been written to 'data_frame.csv' and 'data_frame.json'")

#Libraries
library(httr)
library(jsonlite)
library(readr)

# Fetch JSON data from the API
url <- "https://api.reforestum.com/v1/forests"
response <- GET(url)
json_data <- content(response, "text")

# Parse JSON data
parsed_data <- fromJSON(json_data, flatten = TRUE)

# Convert to dataframe
forests_df <- as.data.frame(parsed_data)

# Print the dataframe
print(forests_df)


# Convert dataframe to JSON and save to file
json_output <- toJSON(forests_df, pretty = TRUE)
write(json_output, file = "forests.json")

# Convert dataframe to CSV and save to file
write_csv(forests_df, "forests.csv")

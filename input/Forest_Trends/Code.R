# Load required libraries
library(jsonlite)

# Define the API endpoint URL
api_url <- "https://www.forest-trends.org/wp-content/themes/foresttrends/map_tools/project_fetch.php?ids="

# Make the API request and parse JSON response
response <- jsonlite::fromJSON(api_url)

# Check the structure of the response
str(response)

# Convert the response to a dataframe (assuming it's a list of lists)
df <- as.data.frame(response)

# View the dataframe
print(df)





# Filter for rows where markers.type is "Forest and land-use carbon"
reforestation_data <- subset(df, markers.type == "Forest and land-use carbon")


# Write DataFrame to CSV
write.csv(df, file = "data.csv", row.names = FALSE)


# Write DataFrame to JSON
write_json(df, "data.json")

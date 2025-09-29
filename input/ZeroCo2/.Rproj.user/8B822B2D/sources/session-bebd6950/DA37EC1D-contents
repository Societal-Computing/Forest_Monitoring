library(httr)
library(jsonlite)
library(tidyverse)

# Define the URL
url <- "https://app.zeroco2.eco/api/countryProjects?limit=12&offset=0&orderBy=%5B%7B%22column%22%3A%22sort%22%2C%22order%22%3A%22asc%22%7D%5D&where=%7B%22currentLang%22%3A%22en%22%7D"

# Fetch the JSON data
response <- GET(url)

# Check if the request was successful
if (status_code(response) == 200) {
  # Parse the JSON content
  json_data <- content(response, as = "text")
  parsed_data <- fromJSON(json_data, flatten = TRUE)
  
  # Convert the parsed data to a dataframe
  df <- as.data.frame(parsed_data)
  
  # Unnest the projects list-column
  projects_df <- df %>%
    select(countryId, countryName, projects) %>%
    unnest(projects)
  
  # Further unnest nested columns within projects
  projects_cleaned <- projects_df %>%
    unnest_wider(description_translations) %>%
    unnest_wider(project_name_translations) %>%
    unnest_wider(gps_position.coordinates, names_sep = "_") %>%
    unnest_wider(gps_position.type, names_sep = "_")
  
  # Print the cleaned dataframe
  print(projects_cleaned)
} else {
  print(paste("Request failed with status code:", status_code(response)))
}

# Convert the dataframe to JSON
projects_json <- toJSON(projects_df, pretty = TRUE)

# Print the JSON string
print(projects_json)

# Optionally, write the JSON to a file
write(projects_json, file = "ZeroCo2.json")
} else {
  print(paste("Request failed with status code:", status_code(response)))
}
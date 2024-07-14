# Install and load required packages
install.packages("dplyr")
install.packages("jsonlite")

library(dplyr)
library(jsonlite)

# Create a dataframe with the extracted information
data <- data.frame(
  Coordinates = c(
    "53.11647, 11.51989", "53.11342, 11.52117", "53.11342, 11.52117", "49.37475, 9.09272", "49.45408, 9.04006", 
    "49.45333, 9.04072", "49.45408, 9.04006", "49.4535, 9.0366", "49.4528, 9.0407", "49.4543, 9.0596", 
    "49.6373, 9.5352", "49.6282, 9.5118", "49.5576, 9.4918", "49.5927, 9.5377"
  ),
  Location = c(
    "Lenzen/Ausbau Sterbitz", "Am Rudower See", "Am Rudower See", "Binau Tannicklings", "Eberbach-Unterdielbach Hardtacker",
    "Eberbach-Unterdielbach Hardtacker", "Eberbach-Unterdielbach Hardtacker", "Eberbach-Unterdielbach Hard",
    "Eberbach-Unterdielbach Hardtacker", "Waldbrunn Stickbaurenwald", "Hardheim Breitenbel", "Hardheim Marle",
    "Hardheim-Erfeld Kehrenweg-Rain", "Hardheim-Erfeld Kappel"
  ),
  Area_m2 = c(113470, 42114, 378804, 17035, 15136, 5053, 5043, 5218, 5017, 12044, 4850, 8134, 7474, 9746),
  Number_of_Planted_Trees = c(21000, 7000, NA, 2000, 100, 10, 70, 160, NA, 1500, 0, 200, 900, 1000),
  Protected_Natural_Area_m2 = c(113470, 42114, NA, 17035, 15136, 5053, 5043, 5218, 5017, 12044, 4850, 8134, 7474, 9746),
  Description = c(
    "The breath of the woods. When trees breathe, they absorb large amounts of CO₂. For this reason, we protect the trees and turn this extensively cultivated logging site into future woodland! We are reforesting this biosphere reserve site, adding more tree species to this former monoculture. Of course, our trees may grow as tall as they want for all their lives. This creates a habitat for maximum biodiversity: a protected shelter for wildlife and nature.",
    "Underneath a green sky. Say yes to wilderness: Along the banks of this lake, a habitat for wildlife and nature is coming about. This idyllic site has already become home to a white-tailed eagle. We are adding various tree species to the monoculture, creating a protected shelter for endangered plants and animals. This will provide a wider variety of ecological niches.",
    "Diese Fläche am Rudower See ist unser Zukunftsprojekt. Ein weiterer Lebensraum für Wildtiere und Natur. Auch diese Fläche liegt im Biosphärenreservat Flusslandschaft Elbe-Brandenburg und bietet daher die besten Voraussetzungen für unser Urwaldprojekt. Unser Ziel für das kommende Jahr ist der Kauf dieser etwa 400.000 qm Land. Dazu brauchen wir auch eure wertvolle Unterstützung.",
    "Tomorrow's jungle - in full bloom today. In Binau, we have turned wasteland back into flourishing nature. By planting over 2,000 trees and revegetating alkaline grassland, we created a new habitat for nature - and a home for deer, wild boars, fire salamanders, lizards, wild bees, bumblebees and butterflies. There is water supply from a picturesque spring, and there are 25 different species of trees contributing not only to maximum biodiversity but also to a great variety of food sources.",
    "To Bee or not to Bee: in Eberbach-Unterdielbach, that's a simple question. We have converted intensively cultivated farmland into a wildflower meadow: a little paradise for wild bees, bumblebees, butterflies and birds. Of course, we used only certified native seeds. We rounded out this protected habitat with fruit-bearing trees, shrubs and a hedge made from dead wood. The bees love it ... and so do all the other wild animals.",
    "Earth laughs in flowers. That's what the poet Ralph Waldo Emerson said. We are not going to contradict - instead we have transformed intensively cultivated farmland back into colorful nature to put a smile on everyone's face. We created a protected habitat for wild bees, bumblebees, butterflies and birds. And frogs, newts and other amphibians feel at home here as well: We planted alders and left natural biotopes to ensure ideal living conditions for them. Artificial water drains were removed, allowing the soil to serve as a natural water reservoir once again.",
    "Today I didn't pick flowers for you – so I can give you their life. We took inspiration from these words by German poet Christian Morgenstern and turned farmland near Eberbach into a meadow with wildflowers, shrubs and trees. Of course, no flowers must be picked from this meadow. With your support, we give life not only to the wildflowers, but also to wild bees, bumblebees, butterflies and birds!",
    "Pure nature. A perfect idyll in the great outdoors: We planted 160 trees of various native species on a meadow at the edge of the woodland and also added a biotope. By acquiring the land, the area is permanently protected: a wonderful and safe place for plants and animals.",
    "Diese Wiese liegt am Waldrand und schließt die Lücke zu den anderen Flächen. Die Nähe zum Biotop ermöglicht einen besonderen Lebensraum. Die Fläche ist durch den Erwerb dauerhaft geschützt.",
    "The future belongs to this forest. Waldbrunn lies at the foot of a dormant volcano, the Katzenbuckel, which translates to ‘Cat’s Hump’. With your support, we bought the local wood and prevented further agricultural exploitation. We also planted 1,500 trees to boost biodiversity. The woodland can now grow undisturbed to become tomorrow’s jungle. This allows the site to protect the climate and serve as a home for plants and animals.",
    "In the mood for wood. Trees against climate change: Starting 2023, we will turn farmland into a place of maximum biodiversity, thanks to your support. So you can enjoy nature here with all your senses.",
    "Forest are true miracles. The climate killer CO2 is an elixir of life for trees: They transform it into carbon and store it. Starting in 2023, we will create a shelter for nature and wildlife here – a protected habitat that will be planted with sustainability in mind. All of this is made possible by your donations: Thank you very much!",
    "Christmas comes but once a year – but climate change is permanent! For this reason, we are converting a former Christmas tree farm into a future forest. By the end of 2023, we will have reforested this site with 1,500 trees from various species, creating a protected habitat for animals and plants.",
    "Biodiversity instead of monoculture. Christmas tree farms are monocultures. They are not only harmful to the soil, but they also deprive wildlife of their natural habitat. This is bad for the environment! In order to give nature something back, we bought the former Christmas tree farm at this site. We are turning it into a future forest, a climate forest, which may grow freely and without any human intervention – for all its life."
  )
)

# Print the dataframe
print(data)

# Save the dataframe to CSV and JSON files
write.csv(data, "projects_data.csv", row.names = FALSE)
write_json(data, "projects_data.json", pretty = TRUE)

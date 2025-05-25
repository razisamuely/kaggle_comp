# Create data folder with subfolders for each competition
mkdir -p data/{smoker-status,blueberry,cirrhosis,mohs}

# Download and extract each competition to its own subfolder
kaggle competitions download -c apl-2025-spring-smoker-status -p data/smoker-status/
cd data/smoker-status && unzip apl-2025-spring-smoker-status.zip && rm *.zip && cd ../..

kaggle competitions download -c apl-2025-spring-blueberry -p data/blueberry/
cd data/blueberry && unzip apl-2025-spring-blueberry.zip && rm *.zip && cd ../..

kaggle competitions download -c apl-2025-spring-cirrhosis -p data/cirrhosis/
cd data/cirrhosis && unzip apl-2025-spring-cirrhosis.zip && rm *.zip && cd ../..

kaggle competitions download -c apl-2025-spring-mohs -p data/mohs/
cd data/mohs && unzip apl-2025-spring-mohs.zip && rm *.zip && cd ../..

# Show organized structure
echo "Data folder structure:"
tree data/ || find data/ -type f
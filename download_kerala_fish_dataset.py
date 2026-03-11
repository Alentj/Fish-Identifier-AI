import os
from icrawler.builtin import BingImageCrawler

# Base dataset folder
BASE_DIR = "kerala_fish_dataset"

# Kerala fish species list
fish_species = {
    "sardine": "Sardinella longiceps fish",
    "mackerel": "Rastrelliger kanagurta fish",
    "seer_fish": "Scomberomorus commerson fish",
    "anchovy": "Stolephorus indicus fish",
    "karimeen": "Etroplus suratensis fish",
    "tuna": "Thunnus albacares fish",
    "pomfret": "Pampus argenteus fish",
    "barracuda": "Sphyraena barracuda fish",
    "trevally": "Caranx ignobilis fish",
    "milkfish": "Chanos chanos fish",
    "catfish": "Clarias batrachus fish",
    "tilapia": "Oreochromis niloticus fish",
    "ribbon_fish": "Trichiurus lepturus fish",
    "grouper": "Epinephelus malabaricus fish",
    "shark": "Carcharhinus limbatus fish",
    "silver_belly": "Leiognathus splendens fish",
    "threadfin_bream": "Nemipterus japonicus fish",
    "stingray": "Dasyatis fish",
    "croaker": "Johnius dussumieri fish",
    "sole_fish": "Cynoglossus fish"
}

# Create main dataset directory
os.makedirs(BASE_DIR, exist_ok=True)

for folder, keyword in fish_species.items():

    save_path = os.path.join(BASE_DIR, folder)

    print(f"\n📥 Downloading images for: {folder}")

    os.makedirs(save_path, exist_ok=True)

    try:
        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": save_path}
        )

        crawler.crawl(
            keyword=keyword,
            max_num=400,
            min_size=(200, 200)
        )

    except Exception as e:
        print(f"⚠ Error downloading {folder}: {e}")

print("\n✅ Kerala fish dataset download completed!")
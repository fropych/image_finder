from pathlib import Path
import re
import aiohttp
import asyncio
import aiofiles
import hydra
from bs4 import BeautifulSoup
from omegaconf import DictConfig
import pyrootutils
from tqdm.asyncio import tqdm_asyncio
import pandas as pd

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


url_base = "https://imgflip.com"
unique_filenames = set()


async def image_download(session, url, path):
    if path.exists():
        return False
    async with session.get(url) as response:
        img = await response.content.read()

    async with aiofiles.open(path, "wb") as img_file:
        await img_file.write(img)
    return True


async def proccess_page(session, html, image_data, path_to_save):
    soup_main = BeautifulSoup(html, features="html.parser")

    for preview_img in soup_main.find_all("img", {"class": "shadow"}):
        parent = preview_img.parent

        if not parent.parent.find("div", {"class": "mt-animated-label"}):
            async with session.get(f"{url_base}{parent.get('href')}") as response:
                html_sub = await response.text()
            soup_sub = BeautifulSoup(html_sub, features="html.parser")

            template_name = soup_sub.find("h1").text[12:]
            template_name = re.sub(r"[^A-Za-z ]", "", template_name).lower().strip()
            if template_name in unique_filenames:
                continue
            unique_filenames.add(template_name)
            template_img_url = (
                soup_sub.find("a", {"class": "meme-link"}).find("img").get("src")
            )
            if template_img_url.startswith("//i.imgflip"):
                template_img_url = f"https:{template_img_url}"
            else:
                template_img_url = f"{url_base}{template_img_url}"
            template_filename = f"{template_name}.jpg"
            template_path = path_to_save / template_filename

            await image_download(session, template_img_url, template_path)

            image_data.append(
                {
                    "name": template_name,
                    "isTemplate": True,
                    "exampleId": None,
                    "exampleText": None,
                    "filename": template_filename,
                    "url": template_img_url,
                }
            )

            for i, img in enumerate(soup_sub.find_all("img", {"class": "base-img"})):
                example_img_url = f"https:{img.get('src')}"
                example_filename = f"{template_name}_{i}.jpg"
                example_path = path_to_save / example_filename
                text = img.get("alt").replace("\n", "")

                await image_download(session, example_img_url, example_path)

                image_data.append(
                    {
                        "name": template_name,
                        "isTemplate": False,
                        "exampleId": i,
                        "exampleText": text,
                        "filename": example_filename,
                        "url": example_img_url,
                    }
                )


async def parse(path_to_save):
    async with aiohttp.ClientSession() as session:
        path_to_save.mkdir(parents=True, exist_ok=True)

        tasks = []
        image_data = []
        for page in range(30):
            async with session.get(
                f"{url_base}/memetemplates?sort=top-all-time&page={page}"
            ) as response:
                html = await response.text()
            task = asyncio.create_task(
                proccess_page(session, html, image_data, path_to_save)
            )
            tasks.append(task)
        await tqdm_asyncio.gather(*tasks)

    return image_data


async def async_main(csv_dir, raw_images_dir):
    image_data = await parse(raw_images_dir)

    df = pd.DataFrame(image_data)
    df.to_csv(csv_dir / "raw_images.csv", encoding="utf-8", index=False)


def main(cfg: DictConfig):
    csv_dir = Path(cfg.paths.csv_dir.raw)
    raw_images_dir = Path(cfg.paths.images_dir.raw)
    asyncio.run(async_main(csv_dir, raw_images_dir))


if __name__ == "__main__":
    main()

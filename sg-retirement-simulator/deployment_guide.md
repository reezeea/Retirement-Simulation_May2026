# How to Publish and Run Your App on Streamlit Community Cloud

Since Python isn't currently set up on your computer, the easiest way to run and share this app is to publish it to **GitHub** and deploy it using **Streamlit Community Cloud** (a free hosting service for Streamlit apps).

Here is the step-by-step guide to get your app online:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com/) and sign up for a free account if you don't have one.
2. Once logged in, click the **"+" icon** in the top right corner and select **"New repository"**.
3. Give your repository a name (e.g., `sg-retirement-simulator`).
4. Choose whether to make it **Public** (anyone can see the code) or **Private** (only you can see it). Streamlit Community Cloud allows free deployments for both.
5. **Important:** Do NOT check the boxes for "Add a README file" or ".gitignore" right now (you already have these files locally).
6. Click the green **"Create repository"** button.

## Step 2: Upload Your Files to GitHub

Since you don't have Git installed on your computer, you can upload the files directly through the browser:

1. On your new repository page, look for the link that says **"uploading an existing file"** (it's usually near the top in a section titled "Quick setup" or "Get started by creating a new file or uploading an existing file").
2. Click that link.
3. Open your project folder (`c:\Users\rijia\OneDrive\Desktop\sg-retirement-simulator`) on your computer.
4. **Drag and drop** the following files and folders into the GitHub upload area:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `LICENSE`
   - `src` folder
   - `tests` folder
5. Scroll down to the "Commit changes" section and click the green **"Commit changes"** button.

## Step 3: Deploy to Streamlit Community Cloud

Now that your code is on GitHub, you can deploy it so you can actually use the tool from anywhere!

1. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign up.
2. Select **"Continue with GitHub"** to link your GitHub account.
3. Once logged in, click the **"New app"** button.
4. If asked, click **"Authorize Streamlit"** to give it access to your GitHub repositories.
5. On the "Deploy an app" page, fill in the details:
   - **Repository:** Select the repository you just created (e.g., `yourusername/sg-retirement-simulator`).
   - **Branch:** `main` (or `master`).
   - **Main file path:** Type `app.py`.
   - **App URL:** You can customize the URL if you want (e.g., `my-sg-retirement.streamlit.app`).
6. Click **"Deploy!"**

Streamlit will now start building your app. It will read your `requirements.txt` file, install all the necessary packages (like `pandas`, `plotly`, `numpy`), and launch your app. 

> [!TIP]
> The very first time it deploys, it might take a few minutes as it installs all the dependencies. Once it's done, you'll see your live application, and you can share the URL with anyone!

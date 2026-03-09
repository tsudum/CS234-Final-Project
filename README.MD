# MineRL Setup Guide (Apple Silicon / M1 Mac)

This guide documents the complete process required to run MineRL (v1.x) on an Apple Silicon Mac (M1/M2).

---

## 1. Create Conda Environment

```bash
conda create -n minerl python=3.9
conda activate minerl
```

---

## 2. Install Java 8 (Temurin)

MineRL requires JDK 8.

Install:

```bash
brew install --cask temurin@8
```

Verify installation:

```bash
ls /Library/Java/JavaVirtualMachines
```

You should see something like:

```
temurin-8.jdk
```

---

## 3. Force System to Use JDK 8

Set JAVA_HOME explicitly:

```bash
export JAVA_HOME="/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home"
export PATH="$JAVA_HOME/bin:$PATH"
hash -r
```

Verify:

```bash
java -version
javac -version
which java
which javac
```

Both must report version 1.8.x.

---

## 4. Force Gradle to Use JDK 8

Create or overwrite:

```bash
mkdir -p ~/.gradle
printf "org.gradle.java.home=%s\n" "$JAVA_HOME" > ~/.gradle/gradle.properties
```

This prevents Gradle from using Java 17.

---

## 5. Clone MineRL

```bash
git clone https://github.com/minerllabs/minerl.git
cd minerl
```

---

## 6. Update LWJGL to 3.3.1 (Apple Silicon Fix)

Edit:

```
scripts/mcp_patch.diff
```

Replace all occurrences of:

```
3.2.1
```

with:

```
3.3.1
```

You can do this automatically:

```bash
sed -i '' 's/3\.2\.1/3.3.1/g' ./scripts/mcp_patch.diff
```

---

## 7. Install MineRL

From the cloned repo root:

```bash
pip install .
```

If this succeeds, MineRL installs into your conda environment.

---

## 8. macOS Thread Fix

Edit:

```
minerl/MCP-Reborn/launchClient.sh
```

Find:

```bash
java -Xmx$maxMem -jar $fatjar --envPort=$port
```

Change to:

```bash
java -Xmx$maxMem -XstartOnFirstThread -jar $fatjar --envPort=$port
```

Save the file.

---

## 9. Remove macOS GLFW Crash Code

Open:

```
minerl/MCP-Reborn/src/main/java/net/minecraft/client/MainWindow.java
```

### Remove this line:

```java
GLFW.glfwSetWindowIcon(this.handle, buffer);
```

### Empty out the `checkGlfwError` method

Change:

```java
public static void checkGlfwError(...) {
    ...
}
```

To:

```java
public static void checkGlfwError(BiConsumer<Integer, String> glfwErrorConsumer) {
}
```

Do NOT delete the method header. Only remove its contents.

Save the file.

---

## 10. Rebuild Minecraft

```bash
cd minerl/MCP-Reborn
./gradlew clean build shadowJar
```

Wait until the build completes successfully.

---

## 11. Copy Rebuilt MCP into Installed Package

Find where MineRL is installed:

```bash
python -c "import minerl, os; print(os.path.dirname(minerl.__file__))"
```

This prints something like:

```
/Users/yourname/miniconda3/envs/minerl/lib/python3.9/site-packages/minerl
```

Copy rebuilt MCP files into that directory:

```bash
SITEPKG="/path/from/above"
cp -rf ./* "$SITEPKG/MCP-Reborn/"
```

Replace `/path/from/above` with the actual printed path.

---

## 12. IMPORTANT: Do Not Run From Repo Folder

Do NOT run Python from inside the cloned MineRL repo folder:

```
Final-Project/minerl
```

If you do, Python will import the cloned source instead of the installed package.

Always run from another directory, for example:

```bash
cd ~
```

---

## 13. Test Environment

Create a file called `test_minerl.py`:

```python
import gym
import minerl

env = gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()

print("keys:", obs.keys())
print("pov shape:", obs["pov"].shape)

env.close()
```

Run:

```bash
python test_minerl.py
```

Expected output:

```
keys: dict_keys(['pov'])
pov shape: (360, 640, 3)
```

If you see this, MineRL is fully working.

---

## Supported Environments

Once setup works, you can run:

- MineRLBasaltFindCave-v0
- MineRLBasaltMakeWaterfall-v0
- MineRLBasaltBuildVillageHouse-v0
- MineRLBasaltCreateVillageAnimalPen-v0
- MineRLObtainDiamond-v0 (heavier)

---

## Dev

```
conda activate minerl
```

---

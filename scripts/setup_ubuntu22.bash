#!/bin/bash

sudo -v

function ask_yes_no {
    local prompt="$1"
    local default="$2"
    read -p "$prompt " answer
    if [[ $answer == "" ]]; then
        answer="$default"
    fi
    case ${answer:0:1} in
        y|Y )
            true
            ;;
        * )
            false
            ;;
    esac
}

ask_yes_no "Install CARLA? [Y/n]" "y"
install_carla=$?

ask_yes_no "Make CARLA accessible on the PATH with the name carla? [Y/n]" "y"
add_carla_to_path=$?

ask_yes_no "Install Poetry? [Y/n]" "y"
install_poetry=$?

# Install carla
if [ $install_carla -eq 0 ]; then
    CARLA_VERSION="CARLA_0.9.15"
    CARLA_ARCHIVE="$CARLA_VERSION.tar.gz"
    CARLA_DOWNLOAD_LINK="https://folk.ntnu.no/dagbo/carla/$CARLA_ARCHIVE"
    CARLA_DEST="/usr/local/carla/$CARLA_VERSION"

    echo "Downloading CARLA"
    curl -O -f "$CARLA_DOWNLOAD_LINK"

    echo "Extracting CARLA"
    sudo mkdir -p "$CARLA_DEST"
    sudo tar -xzf "$CARLA_ARCHIVE" -C "$CARLA_DEST"
    rm "$CARLA_ARCHIVE"

    # Make the current user own all folders and files within the Carla folder
    sudo chown -R $USER $CARLA_DEST

    # To ensure CarlaUE4.sh can be run from anywhere, create a symbolic link in /usr/local/bin, a
    # directory in the system's PATH.
    if [ $add_carla_to_path -eq 0 ]; then
        sudo ln -s "$CARLA_DEST/CarlaUE4.sh" /usr/local/bin/carla
    fi
fi

# Install Poetry
if [ $install_poetry ]; then
    echo "Installing Poetry"
    curl -sSL https://install.python-poetry.org | python3 -
fi

echo "Setting up Python enviorment"
poetry install --no-root

echo "Setup done. Activate Python environment with \"poetry shell\""

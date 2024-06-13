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

ask_yes_no "Install additional CARLA maps? [Y/n]" "y"
install_additional_maps=$?

ask_yes_no "Make CARLA accessible on the PATH with the name carla? [Y/n]" "y"
add_carla_to_path=$?

ask_yes_no "Install Poetry? [Y/n]" "y"
install_poetry=$?

# Install CARLA
CARLA_VERSION="CARLA_0.9.15"
CARLA_ARCHIVE="$CARLA_VERSION.tar.gz"
CARLA_DOWNLOAD_LINK="https://folk.ntnu.no/dagbo/carla/$CARLA_ARCHIVE"
CARLA_DEST="/usr/local/carla/$CARLA_VERSION"
if [ $install_carla -eq 0 ]; then
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

# Install additional CARLA maps
if [ $install_additional_maps -eq 0 ]; then
    MAPS_ARCHIVE="AdditionalMaps_0.9.15.tar.gz"
    MAPS_DOWNLOAD_LINK="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/$MAPS_ARCHIVE"
    MAPS_DEST="$CARLA_DEST/Import/$MAPS_ARCHIVE"

    echo "Downloading additional CARLA maps"
    curl -f $MAPS_DOWNLOAD_LINK -o $MAPS_DEST

    # TODO: Delete
    # echo "Extracting CARLA maps"
    # mkdir -p "$MAPS_DEST_FOLDER"
    # tar -xzf "$MAPS_ARCHIVE" -C "$MAPS_DEST_FOLDER"
    # rm "$MAPS_ARCHIVE"

    pushd $CARLA_DEST
        ./ImportAssets.sh
    popd

    # TODO: Delete
    # Make the current user own all folders and files within the Carla folder
    # sudo chown -R $USER $CARLA_DEST
fi

# Install Poetry
if [ $install_poetry -eq 0 ]; then
    echo "Installing Poetry"
    curl -sSL https://install.python-poetry.org | python3 -
fi

echo "Setting up Python enviorment"
poetry install --no-root

echo "Setup done. Activate Python environment with \"poetry shell\""

#!/bin/bash

### Written by Martino Sorbaro, January 2016
### Run it as 'sh setup.sh', no sudo needed
### I'm not responsible for the sudden Bad Things
### that may occur as a consequence of using this.

xmltext='<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
	<mime-type type="application/x-ipynb+json">
		<comment>Ipython Notebook</comment>
		<icon name="application-x-ipynb+json"/>
		<glob-deleteall/>
		<glob pattern="*.ipynb"/>
	</mime-type>
</mime-info>
'

apptext='#!/bin/bash
netstat -tln |grep "8902"
# if not found - equals to 1, start it
if [ $? -eq 1 ]
then
jupyter notebook / --no-browser --port=8902 &
sleep .5
fi
xdg-open "http://localhost:8902/notebooks$1"
'

desktoptext='[Desktop Entry]
Name=ipynb
Exec=bash -c "$HOME/.local/bin/ipynb %U"
MimeType=application/x-ipynb+json
Icon=ipynb
Terminal=false
Version=1.0
Type=Application'

templatetext='{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.4.3+"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}'

# creates relevant folders if not existent
mkdir -p ~/.local/share/mime/packages/
mkdir -p ~/.local/bin/
mkdir -p ~/.local/share/applications/
mkdir -p ~/Templates/

#creates a new MIME type for ipython notebooks
(echo "$xmltext" > ~/.local/share/mime/packages/ipynb.xml && echo "~/.local/share/mime/packages/ipynb.xml created")
update-mime-database ~/.local/share/mime/ && echo "Updated mime database."

#sets the icon for .ipynb files
#sudo cp stuff/application-x-ipynb+json.svg  /usr/share/icons/gnome/scalable/mimetypes/ && echo "Icon copied in /usr/share/icons/gnome/scalable/mimetypes/"
#sudo gtk-update-icon-cache /usr/share/icons/gnome/ -f && echo "Updated icon cache."

#saves the ipynb launcher script in /usr/local/bin
(echo "$apptext" > ~/.local/bin/ipynb && echo "Script copied in ~/.local/bin/")
chmod u+x ~/.local/bin/ipynb

# sets default application for the MIME type to "ipynb"
(echo "$desktoptext" > ~/.local/share/applications/ipynb.desktop && echo ".desktop file added in ~/.local/share/applications/")
#echo application/x-ipynb+json=ipynb.desktop >> ~/.local/share/applications/mimeapps.list
update-desktop-database ~/.local/share/applications && echo "Updated desktop database."

#puts a file in "Templates"
(echo "$templatetext" > ~/Templates/IPythonNotebook.ipynb && echo "Template added.")

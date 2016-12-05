This version includes a new member of the FindBoxByLabel class that finds the text field that most likely corresponds to the label found, and prints all the text in the box to console. I've added a directory, images/windows, that includes 9 screen captures of particular windows of software running in Windows. Images 7-9 were those provided by Jon. After compiling, you can run the program with the command

      ./textboxfinder images/windows/window#.png some textibox label here

where # is any digit from 1-9. Three images will be written to the current directory, one showing the text groupings, one showing the detected rectangles, and one showing the found label and corresponding field/text. The same images will be displayed as well. The actual Tesseract results for the label and text field will be printed to console.

In this version I've removed the DetectText class code that was included three weeks ago, as the program no longer uses the Stroke Width Transform. Instead, you'll find a new FindBoxByLabel class that uses the antialiased-text-based technique.
After building the executable, you can call it using 
     
             ./textboxfinder images/textboxes/test##/screen.png some input query here

where you replace ## with a pair of digits from 01-23 (keep in mind that some screenshots are duplicated, as I tested various templates on each while making the findBoxByTemplate class last month). test23/screen.png is the google homepage image that I used in the slideshow.

The program should print the rutime, queried text, and matched text to console, and will display two images, one with all of the candidate text groups found and one with the matched group boxed.

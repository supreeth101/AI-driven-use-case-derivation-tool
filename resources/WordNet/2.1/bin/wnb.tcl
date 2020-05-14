#!/bin/sh 
# the following line is evaluated by sh but ignored by tcl \
wishwn "$0" "$@" &
# the following line is evaluated by sh but ignored by tcl \
exec true
# the preceding lines make this script self-executing on unix systems

##########################################################################
#                                                                        # 
#  A Tcl/Tk interface for WordNet                                        # 
#  by David Slomin (dgslomin@princeton.edu) - 6/97                       # 
#  based upon the X-Windows version by Brian Gustafson - 5/91            # 
#                                                                        # 
##########################################################################

if {$tcl_platform(platform) == "windows"} {
  package require registry 1.1
}

### Default Configuration

set showcontextualhelp 0 
set showglosses 1
set wordwrap 1
set showfileinfo 0
set showbyteoffset 0
set showsenseflag 0
set maxhistorylength 10
set fontname times
set fontsize 2
set fontsizelist {100 120 140 180 240}

### Indentation constants

# "->"                         # "=>"
set fontpt1offset(times,1) 20; set fontpt2offset(times,1) 20
set fontpt1offset(times,2) 20; set fontpt2offset(times,2) 20
set fontpt1offset(times,3) 26; set fontpt2offset(times,3) 26
set fontpt1offset(courier,1) 23; set fontpt2offset(courier,1) 23
set fontpt1offset(courier,2) 28; set fontpt2offset(courier,2) 28
set fontpt1offset(courier,3) 34; set fontpt2offset(courier,3) 34
set fontpt1offset(helvetica,1) 19; set fontpt2offset(helvetica,1) 19
set fontpt1offset(helvetica,2) 24; set fontpt2offset(helvetica,2) 24
set fontpt1offset(helvetica,3) 27; set fontpt2offset(helvetica,3) 27

# "HAS PART:"		      # "PART OF:"                 
set fonthpoffset(times,1) 66; set fontpooffset(times,1) 58
set fonthpoffset(times,2) 77; set fontpooffset(times,2) 66
set fonthpoffset(times,3) 100; set fontpooffset(times,3) 87
set fonthpoffset(courier,1) 71; set fontpooffset(courier,1) 64
set fonthpoffset(courier,2) 91; set fontpooffset(courier,2) 82
set fonthpoffset(courier,3) 111; set fontpooffset(courier,3) 100
set fonthpoffset(helvetica,1) 70; set fontpooffset(helvetica,1) 62
set fonthpoffset(helvetica,2) 80; set fontpooffset(helvetica,2) 70
set fonthpoffset(helvetica,3) 102; set fontpooffset(helvetica,3) 90

# "HAS MEMBER:"               # "MEMBER OF:"
set fonthmoffset(times,1) 74; set fontmooffset(times,1) 81
set fonthmoffset(times,2) 85; set fontmooffset(times,2) 92
set fonthmoffset(times,3) 108; set fontmooffset(times,3) 118
set fonthmoffset(courier,1) 75; set fontmooffset(courier,1) 79
set fonthmoffset(courier,2) 96; set fontmooffset(courier,2) 102
set fonthmoffset(courier,3) 118; set fontmooffset(courier,3) 124
set fonthmoffset(helvetica,1) 77; set fontmooffset(helvetica,1) 84
set fonthmoffset(helvetica,2) 87; set fontmooffset(helvetica,2) 95
set fonthmoffset(helvetica,3) 111; set fontmooffset(helvetica,3) 121

# "HAS SUBSTANCE:"             # "SUBSTANCE OF:"
set fonthsoffset(times,1) 105; set fontsooffset(times,1) 97
set fonthsoffset(times,2) 124; set fontsooffset(times,2) 113
set fonthsoffset(times,3) 157; set fontsooffset(times,3) 146
set fonthsoffset(courier,1) 106; set fontsooffset(courier,1) 98
set fonthsoffset(courier,2) 136; set fontsooffset(courier,2) 127
set fonthsoffset(courier,3) 166; set fontsooffset(courier,3) 155
set fonthsoffset(helvetica,1) 111; set fontsooffset(helvetica,1) 103
set fonthsoffset(helvetica,2) 127; set fontsooffset(helvetica,2) 117
set fonthsoffset(helvetica,3) 166; set fontsooffset(helvetica,3) 156

# " "			     # "[0-9]"
set fontspoffset(times,1) 3; set fontNUMoffset(times,1) 6
set fontspoffset(times,2) 3; set fontNUMoffset(times,2) 7
set fontspoffset(times,3) 4; set fontNUMoffset(times,3) 10
set fontspoffset(courier,1) 7; set fontNUMoffset(courier,1) 8
set fontspoffset(courier,2) 9; set fontNUMoffset(courier,2) 11
set fontspoffset(courier,3) 11; set fontNUMoffset(courier,3) 13
set fontspoffset(helvetica,1) 4; set fontNUMoffset(helvetica,1) 7
set fontspoffset(helvetica,2) 4; set fontNUMoffset(helvetica,2) 8
set fontspoffset(helvetica,3) 5; set fontNUMoffset(helvetica,3) 9

# ". "
set fontsp2offset(times,1) 6;
set fontsp2offset(times,2) 6;
set fontsp2offset(times,3) 7;
set fontsp2offset(courier,1) 12;
set fontsp2offset(courier,2) 15;
set fontsp2offset(courier,3) 19;
set fontsp2offset(helvetica,1) 7;
set fontsp2offset(helvetica,2) 7;
set fontsp2offset(helvetica,3) 11;

set labonly 0
set version "2.1"

if {$tcl_platform(platform) == "unix"} {
   if {[lsearch -exact [array names env] WNHOME] == -1} {
	set resourcedir "/usr/local/WordNet-2.1/lib/wnres"
   } else {
	set resourcedir "$env(WNHOME)/lib/wnres"
   }
   set configfile "$env(HOME)/.wnrc"
   if [ file exists $configfile ] {
       source $configfile
   }
   set printcommand "lpr"
}
if {$tcl_platform(platform) == "windows"} {
    set UserRegPath "HKEY_CURRENT_USER\\Software\\WordNet\\2.1"
    if { [registry values $UserRegPath "wnrc" ] != "" } {
	eval [ registry get $UserRegPath "wnrc" ]
    }
    if { [registry values $UserRegPath "WNHome"] != "" } {
	 append resourcedir [registry get $UserRegPath "WNHome"] "/lib/wnres"
    } else {
    	 set MachineRegPath "HKEY_LOCAL_MACHINE\\Software\\WordNet\\2.1"
    	 if { [registry values $MachineRegPath "WNHome" ] != "" } {
     	    append resourcedir [registry get $MachineRegPath "WNHome"] "/lib/wnres"
	 } else {
          set resourcedir "C:/Program Files/WordNet/2.1/lib/wnres"
       }
    }
}

### Startup

wm title . "WordNet 2.1 Browser"

### Primary Functions

proc displayvalidsearchmenus {searchword} {
   global posmenu
   blackout 1
   set gotone 0
   pack forget .wordframe.overview
   for {set posnumber 1} {$posnumber <= 4} {incr posnumber} {
      set pos [lindex {noun verb adj adv} [expr $posnumber - 1]]
      pack forget .posmenubar.$pos
      set bitfield [findvalidsearches [fixword $searchword] $posnumber]
      if {$bitfield != 0} {
         pack \
            .posmenubar.$pos \
            -side left \
            -padx 2 \
            -pady 2 \
            -before .posmenubar.senselabel
         .posmenubar.$pos.menu delete 0 end
         foreach line $posmenu($posnumber) {
            set label [lindex $line 0]
            regsub -nocase "this" $label $searchword label
            set searchtypenumber [lindex $line 1]
            set abssearchtypenumber [expr abs($searchtypenumber)]
            set action "\
               searchanddisplay \"$searchword\" \"\$g_senses\" $posnumber \
                  $searchtypenumber; \
               history_add \"$searchword\" \"\$g_senses\" $posnumber \
                  $searchtypenumber; \
               pack .wordframe.overview \
                  -side right \
                  -padx 5 \
                  -pady 1
            "
            if {[expr $bitfield & [bit $abssearchtypenumber]]} {
               .posmenubar.$pos.menu add command \
                  -label $label \
                  -command $action
            }
         }
         set gotone 1
      }
   }
   if {!$gotone} {
      .statusbar.status configure \
         -text "No matches found."
   }
   blackout 0
   return $gotone
}

proc displayoverview {searchword} {
   global showfileinfo
   global showbyteoffset
   global showsenseflag
   global wordwrap
   global fontNUMoffset
   global fontsize
   global fontname
   global fontsp2offset

   set fontNUMsize $fontNUMoffset($fontname,$fontsize)	
   set fontsp2size $fontsp2offset($fontname,$fontsize)

   if {$showfileinfo == 2} {fileinfo 1} else {fileinfo 0}
   if {$showbyteoffset == 2} {byteoffset 1} else {byteoffset 0}
   if {$showsenseflag == 2} {senseflag 1} else {senseflag 0}
   blackout 1
   .results.text configure \
      -state normal
   .results.text delete 1.0 end
   pack forget .wordframe.overview
   set gotone 0
   for {set posnumber 1} {$posnumber <= 4} {incr posnumber} {
      set bitfield [findvalidsearches [fixword $searchword] $posnumber]
      if {$bitfield != 0} {
         # The 31 in the following line should correspond to the value of
         # OVERVIEW defined in wnconsts.h in the WordNet library
         set buf [search [fixword $searchword] $posnumber 31 -1]
         set morphedword ""
	 set numspaces 0
	 set iter 0
         set buflines [split $buf "\n"]
         foreach line $buflines {
	    .results.text mark set begofline \
		[.results.text index "end - 2 char"]
            .results.text insert end "$line\n"
            regexp -nocase -- \
               "The (noun|verb|adj|adv) (.*) has \[0-9\]+ senses?" \
               $line dummy dummy morphedword
            set index [string first " --" $line]
            if {$index != -1} {
               set line [string range $line 0 [expr $index - 1]]
            }
            if {[regexp -indices -nocase -- \
               "\[\})>,\] ($morphedword)\[0-9\]*(#\[0-9\]+)?(,|\$)" \
               $line dummy indices]} {
               .results.text tag add overviewhighlight \
                  "end - 2 lines linestart + [lindex $indices 0] chars" \
                  "end - 2 lines linestart + [expr \
                     [lindex $indices 1] + 1] chars"
            } elseif {[regexp -indices -nocase -- \
               "\[0-9\]+\. ($morphedword)\[0-9\]*(#\[0-9\]+)?(,|\$)" \
               $line dummy indices]} {
               .results.text tag add overviewhighlight \
                  "end - 2 lines linestart + [lindex $indices 0] chars" \
                  "end - 2 lines linestart + [expr \
                     [lindex $indices 1] + 1] chars"
	    }
	    set indent [string first "." $line]
	    if {$indent != -1} {
		set numspaces [expr $indent * $fontNUMsize + $fontsp2size]
            } else {set numspaces 0}
	    .results.text mark set endofline [.results.text index \
	     "end - 1 char"]
	    .results.text tag configure indent$posnumber$iter \
		-lmargin2 $numspaces
            .results.text tag add indent$posnumber$iter \
		  begofline endofline
	    .results.text mark set begofline [.results.text index \
	     "end - 2 char"]
	    incr iter
         }
         .results.text delete "end - 1 lines" end
         set gotone 1
      }
   }
   .results.text configure \
      -state disabled
   if {$gotone} { 
      .statusbar.status configure \
         -text "Overview of $searchword" 
   } 
   blackout 0
   setwordwrap $wordwrap
   focus .results.text
   return $gotone
}

proc searchanddisplay {searchword senses posnumber searchtypenumber} {
   global posmenu
   global showcontextualhelp
   global showglosses
   global showfileinfo
   global showbyteoffset
   global showsenseflag
   blackout 1
   glosses $showglosses
   if {$showfileinfo > 0} {fileinfo 1} else {fileinfo 0}
   if {$showbyteoffset > 0} {byteoffset 1} else {byteoffset 0}
   if {$showsenseflag > 0} {senseflag 1} else {senseflag 0}
   set longpos [lindex {noun verb adjective adverb} [expr $posnumber - 1]]
   .results.text configure \
      -state normal
   .results.text delete 0.0 end
   .statusbar.status configure \
      -text "Searching... (press escape to abort)"
   update idletasks
   if {$showcontextualhelp} {
      foreach line $posmenu($posnumber) {
         if {[lindex $line 1] == $searchtypenumber} {
            set linenumber [lsearch $posmenu($posnumber) $line]
         }
      }
      .results.text insert end "\n[contextualhelp $posnumber $linenumber]"
      .results.text mark set endofhelp [.results.text index "end - 1 char"]
      .results.text tag add helpstyle 1.0 endofhelp
   }
   foreach sense [getsenselist $senses] {

	global fontsize

	global fontpt1offset
	global fontpt2offset
	global fonthpoffset
	global fontpooffset
	global fonthmoffset
	global fontmooffset
	global fonthsoffset
	global fontsooffset
	global fontspoffset

	global fontname
	global wordwrap

      if {$wordwrap} {	  
	set searchlines [split [search [fixword $searchword] $posnumber \
         $searchtypenumber $sense] "\n"]
	.results.text mark set begofline [.results.text index "end - 2 char"]

	### Set indentation levels ###	

	set fontpt1size $fontpt1offset($fontname,$fontsize)
	set fontpt2size $fontpt2offset($fontname,$fontsize)
	set fonthpsize $fonthpoffset($fontname,$fontsize)
	set fontposize $fontpooffset($fontname,$fontsize)
	set fonthmsize $fonthmoffset($fontname,$fontsize)
	set fontmosize $fontmooffset($fontname,$fontsize)
	set fonthssize $fonthsoffset($fontname,$fontsize)
	set fontsosize $fontsooffset($fontname,$fontsize)
	set fontspsize $fontspoffset($fontname,$fontsize)

        .statusbar.status configure \
            -text "Formatting... (press escape to abort)"

	set iter 0
	set morphedword ""
	set mword ""
	foreach i $searchlines {
	    .results.text insert end $i\n
	    .results.text mark set endofline [.results.text index \
	     "end - 1 char"]

# added by RIT
	    regexp -nocase -- \
		"\[0-9\]+ (senses|sense) of (.+)" \
		    $i dummy dummy mword
	    set morphedword [string trimright $mword]
   	    set index [string first " --" $i]
	    set schar [string first " " $i]
	    if {$index != -1 && $schar != 0} {
		set line [string range $i 0 [expr $index - 1]]
		if {[regexp -indices -nocase -- \
			 "(\[\.\})>,\] |^|, )($morphedword)\[0-9\]*(#\[0-9\]+)?( \[(\]|,|$\)" \
		    $line dummy dummy indices]} {
		    .results.text tag add overviewhighlight \
			"end - 2 lines linestart + [lindex $indices 0] chars" \
			"end - 2 lines linestart + [expr \
                     [lindex $indices 1] + 1] chars"
		}
            }

	    set numspaces 0
	    set idx [string first "->" $i]
	    if {$idx != -1} {set numspaces \
	       [expr $fontpt1size + $idx * $fontspsize - 1]} \
	     else {set idx [string first "=>" $i]
	      if {$idx != -1} {set numspaces \
		 [expr $fontpt2size + $idx * $fontspsize]} \
              else {set idx [string first "HAS PART:" $i]
	       if {$idx != -1} {set numspaces \
		  [expr $fonthpsize + $idx * $fontspsize - 1]} \
	       else {set idx [string first "PART OF:" $i]
	        if {$idx != -1} {set numspaces \
		   [expr $fontposize + $idx * $fontspsize - 1]} \
	        else {set idx [string first "HAS MEMBER:" $i]
	         if {$idx != -1} {set numspaces \
	            [expr $fonthmsize + $idx * $fontspsize - 1]} \
	         else {set idx [string first "MEMBER OF:" $i]
	          if {$idx != -1} {set numspaces \
	             [expr $fontmosize + $idx * $fontspsize - 1]} \
	          else {set idx [string first "HAS SUBSTANCE:" $i]
		   if {$idx != -1} {set numspaces \
		      [expr $fonthssize + $idx * $fontspsize]} \
	           else {set idx [string first "SUBSTANCE OF:" $i]
	            if {$idx != -1} {set numspaces \
	               [expr $fontsosize + $idx * $fontspsize]} \	    
	     } } } } } } }

	    .results.text tag configure indentstyle$iter -lmargin2 $numspaces
	    .results.text tag add indentstyle$iter begofline endofline
	    .results.text mark set begofline [.results.text index \
	     "end - 2 char"]
	    incr iter
	}
      } else {
	.results.text insert end [search [fixword $searchword] $posnumber \
         $searchtypenumber $sense]
      }

      setwordwrap $wordwrap
   }
   foreach line $posmenu($posnumber) {
      if {[lindex $line 1] == $searchtypenumber} {
         set label "\"[lindex $line 0]\" search for $longpos \"$searchword\""
         eval [lindex $line 2]
         break
      }
   }
   .results.text configure \
      -state disabled
   .statusbar.status configure \
      -text $label
   blackout 0
} 

proc history_add {searchword senses posnumber searchtypenumber} {
   global posmenu
   global maxhistorylength

   if {$posnumber == 0} {
      set label "$searchword - Overview"
      set action " \
         set g_searchword \"$searchword\"; \
         set g_senses \"\"; \
         displayvalidsearchmenus \"$searchword\"; \
         displayoverview \"$searchword\"; \
         .posmenubar.label configure \
            -text \"Searches for $searchword:\" \
      "
   } else {
      set longpos [lindex {Noun Verb Adjective Adverb} [expr $posnumber - 1]]
      foreach line $posmenu($posnumber) {
         if {[lindex $line 1] == $searchtypenumber} {
            if {$senses == ""} {
               set label "$searchword - $longpos / [lindex $line 0]"
            } else {
               set label \
                  "$searchword - $longpos / [lindex $line 0] / $senses"
            }
            break
         }
      }
      set action " \
         set g_searchword \"$searchword\"; \
         set g_senses \"$senses\"; \
         displayvalidsearchmenus \"$searchword\"; \
         searchanddisplay \"$searchword\" \"$senses\" $posnumber \
            $searchtypenumber; \
         pack .wordframe.overview \
            -side right \
            -padx 5 \
            -pady 1; \
         .posmenubar.label configure \
            -text \"Searches for $searchword:\" \
      "
   }
   .menubar.history.menu insert 0 command \
      -label $label \
      -command $action
   if {[.menubar.history.menu index end] == $maxhistorylength} {
      .menubar.history.menu delete end
   }
}

proc saveoptions { } {

    global tcl_platform
    global UserRegPath
    global showcontextualhelp
    global showglosses
    global wordwrap
    global showfileinfo
    global showbyteoffset
    global showsenseflag
    global maxhistorylength
    global fontname
    global fontsize
    global fontsizelist
    global configfile

# Under Unix, write options settings to ".wnrc" file in user's 
# home directory.  
# Under Windows, write to "wnrc" key in registry.

    if {$tcl_platform(platform) == "unix"} {
	set c [open $configfile w]
	puts $c "set showcontextualhelp $showcontextualhelp"
	puts $c "set showglosses $showglosses"
	puts $c "set wordwrap $wordwrap"
	puts $c "set showfileinfo $showfileinfo"
	puts $c "set showbyteoffset $showbyteoffset"
	puts $c "set showsenseflag $showsenseflag"
	puts $c "set maxhistorylength $maxhistorylength"
	puts $c "set fontname $fontname"
	puts $c "set fontsize $fontsize"
	puts $c "set fontsizelist \{ $fontsizelist \}"
    } elseif {$tcl_platform(platform) == "windows"} {
	set c ""
	append c "set showcontextualhelp $showcontextualhelp\n"
	append c "set showglosses $showglosses\n"
	append c "set wordwrap $wordwrap\n"
	append c "set showfileinfo $showfileinfo\n"
	append c "set showbyteoffset $showbyteoffset\n"
	append c "set showsenseflag $showsenseflag\n"
	append c "set maxhistorylength $maxhistorylength\n"
	append c "set fontname $fontname\n"
	append c "set fontsize $fontsize\n"
	append c "set fontsizelist \{ $fontsizelist \}\n"
	registry set $UserRegPath "wnrc" $c
    }
}
 
proc showhelpwidget {w filename windowtitle} {
   if {[winfo exist .$w]} {raise .$w; return}
   toplevel .$w
   wm title .$w $windowtitle
   grid \
      [frame .$w.top \
         -relief raised \
         -borderwidth 1] \
      -row 0 \
      -column 0 \
      -sticky nsew
   grid \
      [frame .$w.buttons \
         -relief raised \
         -borderwidth 1] \
      -row 1 \
      -column 0 \
      -sticky nsew
   grid rowconfigure .$w 0 -weight 1
   grid rowconfigure .$w 1 -weight 0
   grid columnconfigure .$w 0 -weight 1
   text .$w.top.text \
      -wrap word \
      -relief sunken \
      -borderwidth 2 \
      -font -adobe-courier-medium-r-normal-*-*-120-*-*-*-*-*-* \
      -state disabled \
      -yscrollcommand ".$w.top.scrolly set" \
      -width 80 \
      -height 25 \
      -background White \
      -foreground Black
   scrollbar .$w.top.scrolly \
      -command ".$w.top.text yview" \
      -relief sunken \
      -width 12
   grid .$w.top.text \
      -row 0 \
      -column 0 \
      -sticky nsew
   grid .$w.top.scrolly \
      -row 0 \
      -column 1 \
      -sticky nsew
   grid rowconfigure .$w.top 0 -weight 1
   grid rowconfigure .$w.top 1 -weight 0
   grid columnconfigure .$w.top 0 -weight 1
   grid columnconfigure .$w.top 1 -weight 0
   pack \
      [button .$w.buttons.dismiss \
         -text "Dismiss" \
         -command "destroy .$w"] \
      -side top \
      -padx 1 \
      -pady 1
   .$w.top.text configure \
      -state normal
   set fileid [open $filename "r"]
   set filetext [read $fileid]
   close $fileid
   regsub -all ".\b" $filetext "" filetext
   .$w.top.text insert end $filetext
   .$w.top.text configure \
      -state disabled
}

proc printtext {whattoprint} {
   # The following line is a hack to get the passed argument to be
   # recognised in bound actions, like button presses.
   global printtext_whattoprint; set printtext_whattoprint $whattoprint
   global tcl_platform
   switch $tcl_platform(platform) {
      unix {
         global printcommand
         toplevel .printtxt
         wm title .printtxt "Print WordNet Results"
         wm transient .printtxt .
         scan [wm geometry .] "%dx%d+%d+%d" geom_h geom_w geom_x geom_y
         wm geometry .printtxt +[expr $geom_x+50]+[expr $geom_y+50]
         wm resizable .printtxt 0 0
         grab set .printtxt
         pack \
            [frame .printtxt.top] \
            [frame .printtxt.bottom] \
            -side top \
            -padx 10 \
            -pady 10
         pack \
            [label .printtxt.top.label \
               -text "Print command:"] \
            [entry .printtxt.top.entry \
               -textvariable printcommand \
               -background White \
               -foreground Black] \
            -side left
         if {$tcl_platform(platform) != "macintosh"} {
            focus .printtxt.top.entry
         }
         pack \
            [button .printtxt.bottom.print \
               -text "Print" \
               -command {
                  set fileId [open "| $printcommand" w]
                  switch $printtext_whattoprint {
                     "main" {
                        puts $fileId [.results.text get 1.0 end]
                     }
                     "grep" {
                        puts $fileId [join \
                           [.grepwidget.results.frame.list \
                           get 0 end] "\n"]
                     }
                     default {}
                  }
                  close $fileId
                  destroy .printtxt
               }] \
            [button .printtxt.bottom.cancel \
               -text "Cancel" \
               -command {
                  destroy .printtxt
               }] \
            -side left
         bind .printtxt <Return> {
            .printtxt.bottom.print flash
            .printtxt.bottom.print invoke
         }
      }
      windows {
         bell
      }
      macintosh {
         bell
      }
   }
}

proc savetext {whattosave} {
   # The following line is a hack to get the passed argument to be
   # recognised in bound actions, like button presses.
   global tcl_platform
   global savetext_whattosave; set savetext_whattosave $whattosave
   toplevel .saveas
   wm title .saveas "Save WordNet Results To File"
   wm transient .saveas .
   scan [wm geometry .] "%dx%d+%d+%d" geom_h geom_w geom_x geom_y
   wm geometry .saveas +[expr $geom_x+50]+[expr $geom_y+50]
   wm resizable .saveas 0 0
   grab set .saveas
   pack \
      [frame .saveas.top] \
      [frame .saveas.bottom] \
      -side top \
      -padx 10 \
      -pady 10
   pack \
      [label .saveas.top.label \
         -text "Filename:"] \
      [entry .saveas.top.entry \
         -textvariable filename \
         -background White \
         -foreground Black] \
      -side left
   if {$tcl_platform(platform) != "macintosh"} {
      focus .saveas.top.entry
   }
   pack \
      [button .saveas.bottom.save \
         -text "Save" \
         -command { 
            if [file exists $filename] {
               toplevel .savewarning
               wm title .savewarning "Wordnet Warning"
               wm transient .savewarning .saveas
               scan [wm geometry .saveas] "%dx%d+%d+%d" \
                  geom_h geom_w geom_x geom_y
               wm geometry .savewarning +[expr $geom_x+50]+[expr $geom_y+50]
               wm resizable .savewarning 0 0
               grab set .savewarning
               pack \
                  [message .savewarning.message \
                     -text "The file \"$filename\" already exists.\
                        Choose \"append\" to add the new search results onto\
                        the end of the old ones.  Choose \"replace\" to\
                        discard the old search results and store the new ones\
                        in their place.  If you choose \"cancel,\" you will\
                        have the opportunity to select a different file." \
                     -width 300] \
                  [frame .savewarning.bottom] \
                  -side top \
                  -padx 10 \
                  -pady 10
               pack \
                  [button .savewarning.bottom.append \
                     -text "Append" \
                     -command {
                        set fileId [open $filename "a"]
                        switch $savetext_whattosave {
                           "main" {
                              puts $fileId [.results.text get 1.0 end]
                           }
                           "grep" {
                              puts $fileId [join \
                                 [.grepwidget.results.frame.list \
                                 get 0 end] "\n"]
                           }
                           default {}
                        }
                        close $fileId
                        destroy .saveas
                        destroy .savewarning
                     }] \
                  [button .savewarning.bottom.replace \
                     -text "Replace" \
                     -command {
                        set fileId [open $filename "w"]
                        switch $savetext_whattosave {
                           "main" {
                              puts $fileId [.results.text get 1.0 end]
                           }
                           "grep" {
                              puts $fileId [join \
                                 [.grepwidget.results.frame.list \
                                 get 0 end] "\n"]
                           }
                           default {}
                        }
                        close $fileId
                        destroy .saveas
                        destroy .savewarning
                     }] \
                  [button .savewarning.bottom.cancel \
                     -text "Cancel" \
                     -command {
                        destroy .savewarning
                     }] \
                  -side left
               bind .savewarning <Return> {
                  .savewarning.bottom.append flash
                  .savewarning.bottom.append invoke
               }
            } else {
               set fileId [open $filename "w"]
               switch $savetext_whattosave {
                  "main" {
                     puts $fileId [.results.text get 1.0 end]
                  }
                  "grep" {
                     puts $fileId [join [.grepwidget.results.frame.list \
                        get 0 end] "\n"]
                  }
                  default {}
               }
               close $fileId
               destroy .saveas
            }
         }] \
      [button .saveas.bottom.cancel \
         -text "Cancel" \
         -command {destroy .saveas}] \
      -side left
   bind .saveas <Return> {
      .saveas.bottom.save flash
      .saveas.bottom.save invoke
   }
}

proc setmaxhistorylength {} {
   global maxhistorylength
   global tcl_platform
   toplevel .histlen
   wm title .histlen "Set Maximum WordNet Browser History Length"
   wm transient .histlen .
   scan [wm geometry .] "%dx%d+%d+%d" geom_h geom_w geom_x geom_y
   wm geometry .histlen +[expr $geom_x+50]+[expr $geom_y+50]
   wm resizable .histlen 0 0
   grab set .histlen
   pack \
      [frame .histlen.f \
         -relief raised \
         -borderwidth 1] \
      -side top \
      -fill both \
      -expand true
   pack \
      [frame .histlen.f.top] \
      [frame .histlen.f.bottom] \
      -side top \
      -padx 10 \
      -pady 10
   pack \
      [label .histlen.f.top.label \
         -text "Maximum history length:"] \
      [entry .histlen.f.top.entry \
         -textvariable maxhistorylength \
         -background White \
         -foreground Black] \
      -side left
   if {$tcl_platform(platform) != "macintosh"} {
      focus .histlen.f.top.entry
   }
   pack \
      [button .histlen.f.bottom.ok \
         -text "Ok" \
         -command {
            if {[.menubar.history.menu index end] >= $maxhistorylength} {
               .menubar.history.menu delete $maxhistorylength end
            }
            destroy .histlen
         }] \
      -side left
   bind .histlen <Return> {
      .histlen.f.bottom.ok flash
      .histlen.f.bottom.ok invoke
   }
}

proc showaboutbox {} {
   global resourcedir
   if {[winfo exist .aboutbox]} {raise .aboutbox; return}
   toplevel .aboutbox
   wm title .aboutbox "About WordNet Browser"
   wm resizable .aboutbox 0 0
   pack \
      [frame .aboutbox.top \
         -relief raised \
         -borderwidth 1] \
      [frame .aboutbox.bottom \
         -relief raised \
         -borderwidth 1] \
      -side top \
      -fill x \
      -ipadx 3 \
      -ipady 3
   pack \
      [frame .aboutbox.top.left \
         -relief flat] \
      [frame .aboutbox.top.right \
         -relief flat] \
      -side left \
      -fill x \
      -ipadx 3 \
      -ipady 3
   pack \
      [label .aboutbox.top.left.icon \
         -bitmap @$resourcedir/wn.xbm] \
      -side left \
      -padx 10 
   pack \
      [label .aboutbox.top.right.text1 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-180-*-*-*-*-*-*" \
         -text "WordNet Browser"] \
      [label .aboutbox.top.right.text2 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-120-*-*-*-*-*-*" \
         -text "A graphical interface to the\nWordNet online lexical\
            database."] \
      [label .aboutbox.top.right.text3 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-120-*-*-*-*-*-*" \
         -text "This Tcl/Tk version by David Slomin and Randee Tengi."] \
      [label .aboutbox.top.right.text4 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-120-*-*-*-*-*-*" \
         -text "Based upon an earlier X Window version by\nBrian Gustafson."] \
      [label .aboutbox.top.right.text5 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-120-*-*-*-*-*-*" \
         -text "Copyright 1991-2005\nPrinceton University Cognitive Science Lab"] \
      [label .aboutbox.top.right.text6 \
         -anchor w \
         -justify left \
         -font "-adobe-helvetica-medium-r-*-*-*-120-*-*-*-*-*-*" \
         -text "All Rights Reserved"] \
      -side top \
      -fill x
   pack \
      [button .aboutbox.bottom.ok \
         -text "Dismiss" \
         -command {destroy .aboutbox}] \
      -side top \
      -padx 3 \
      -pady 2
}

proc grepword {greptype} {
   global g_greptype; set g_greptype $greptype;
   global g_searchword
   global grepstr
   global tcl_platform

   if {[winfo exist .grepwidget]} {raise .grepwidget; return}

   set grepstr $g_searchword

   toplevel .grepwidget
   switch $g_greptype {
       "substring" { wm title .grepwidget "Substring search (grep)"}
       "ending" { wm title .grepwidget "Ending string search (grep)"}
   }
   grid \
      [frame .grepwidget.inputs \
         -relief raised \
         -borderwidth 1] \
      -row 0 \
      -column 0 \
      -sticky nsew
   grid \
      [frame .grepwidget.results \
         -relief raised \
         -borderwidth 1] \
      -row 1 \
      -column 0 \
      -sticky nsew
   grid \
      [frame .grepwidget.status \
         -relief raised \
         -borderwidth 1] \
      -row 2 \
      -column 0 \
      -sticky nsew
   grid \
      [frame .grepwidget.buttons \
         -relief raised \
         -borderwidth 1] \
      -row 3 \
      -column 0 \
      -sticky nsew
   grid rowconfigure .grepwidget 0 -weight 0
   grid rowconfigure .grepwidget 1 -weight 1
   grid rowconfigure .grepwidget 2 -weight 0
   grid rowconfigure .grepwidget 3 -weight 0
   grid columnconfigure .grepwidget 0 -weight 1
   tk_optionMenu .grepwidget.inputs.pos greppos Noun Verb Adjective Adverb
   grid \
      [label .grepwidget.inputs.label \
         -text "Substring:"] \
      -row 0 \
      -column 0 \
      -sticky ew \
      -padx 2 \
      -pady 2
   grid \
      [entry .grepwidget.inputs.word \
         -textvariable grepstr \
         -width 15 \
         -foreground Black \
         -background White] \
      -row 0 \
      -column 1 \
      -sticky ew \
      -padx 2 \
      -pady 2
   if {$tcl_platform(platform) != "macintosh"} {
      focus .grepwidget.inputs.word
   }
   grid \
      .grepwidget.inputs.pos \
      -row 0 \
      -column 2 \
      -sticky ew \
      -padx 2 \
      -pady 2
   grid rowconfigure .grepwidget.inputs 0 -weight 1
   grid columnconfigure .grepwidget.inputs 0 -weight 0
   grid columnconfigure .grepwidget.inputs 1 -weight 1
   grid columnconfigure .grepwidget.inputs 2 -weight 0
   pack \
      [frame .grepwidget.results.frame] \
      -side top \
      -fill both \
      -expand true \
      -padx 8 \
      -pady 8
   grid \
      [listbox .grepwidget.results.frame.list \
         -yscrollcommand ".grepwidget.results.frame.yscroll set" \
         -foreground Black \
         -background White \
         -highlightthickness 0] \
      -row 0 \
      -column 0 \
      -sticky nsew
   grid \
      [scrollbar .grepwidget.results.frame.yscroll \
         -command ".grepwidget.results.frame.list yview" \
         -width 12 \
         -highlightthickness 0] \
      -row 0 \
      -column 1 \
      -sticky nsew
   grid rowconfigure .grepwidget.results.frame 0 -weight 1
   grid columnconfigure .grepwidget.results.frame 0 -weight 1
   grid columnconfigure .grepwidget.results.frame 1 -weight 0
   pack \
      [label .grepwidget.status.label \
         -text ""] \
      -side top
   pack \
      [frame .grepwidget.buttons.frame] \
      -side top
   grid \
      [button .grepwidget.buttons.frame.search \
         -text "Search" \
         -command {
            if {[.grepwidget.buttons.frame.search cget -text] == "Search"} {
               if {[string length $grepstr] >= 3} {
                  .grepwidget.buttons.frame.search configure \
                     -text "Abort"
                  .grepwidget.status.label configure \
                     -text "Searching... (press escape to abort)"
                  bind .grepwidget <Escape> {
                     .grepwidget.buttons.frame.search flash
                     .grepwidget.buttons.frame.search invoke
                  }
                  .grepwidget.results.frame.list delete 0 end
                  update
                  set posnumber [lsearch {{} Noun Verb Adjective Adverb} \
                     $greppos]
                  set clockstart [clock seconds]
		  switch $g_greptype {
		      "substring" {
			  set buf [search [fixword $grepstr] $posnumber 30 0]
		      }
		      "ending" {
			  set buf [fixgrep [search [fixword $grepstr] $posnumber 30 0] [fixword $grepstr]]
		      }
		  }
                  set clockstop [clock seconds]
                  set buflines [split $buf \n]
                  foreach line $buflines {
                     .grepwidget.results.frame.list insert end $line
                  } 
                  .grepwidget.buttons.frame.search configure \
                     -text "Search"
                  .grepwidget.status.label configure -text ""
                  bind .grepwidget <Escape> {}
               } else {
                  bell
	          set line "Error: search string less than 3 characters\n"
		  .grepwidget.results.frame.list delete 0 end
	          .grepwidget.results.frame.list insert end $line
               }
            } else {
               bell
               abortsearch
            }
         }] \
      -row 0 \
      -column 0 \
      -sticky nsew \
      -padx 2 \
      -pady 2
   grid \
      [button .grepwidget.buttons.frame.save \
         -text "Save" \
         -command "savetext grep"] \
      -row 0 \
      -column 1 \
      -sticky nsew \
      -padx 2 \
      -pady 2

# Printing only works on Unix systems, so don't offer the option on
# other platforms.
   if {$tcl_platform(platform) == "unix"} {
      grid \
         [button .grepwidget.buttons.frame.print \
            -text "Print" \
            -command "printtext grep"] \
         -row 0 \
         -column 2 \
         -sticky nsew \
         -padx 2 \
         -pady 2
   }
   grid \
      [button .grepwidget.buttons.frame.dismiss \
         -text "Dismiss" \
         -command "destroy .grepwidget"] \
      -row 0 \
      -column 3 \
      -sticky nsew \
      -padx 2 \
      -pady 2
   grid rowconfigure .grepwidget.buttons.frame 0 -weight 0
   grid columnconfigure .grepwidget.buttons.frame 0 -weight 1
   grid columnconfigure .grepwidget.buttons.frame 1 -weight 1
   grid columnconfigure .grepwidget.buttons.frame 2 -weight 1
   grid columnconfigure .grepwidget.buttons.frame 3 -weight 1
   bind .grepwidget.results.frame.list <ButtonRelease-1> {
      set chosengrep [.grepwidget.results.frame.list curselection]
      if {$chosengrep != {}} {
         set chosengrepword [.grepwidget.results.frame.list get \
            [lindex $chosengrep 0]]
         set g_searchword $chosengrepword
         set g_senses {}
         golookup
      }
   }
   bind .grepwidget <Return> {
      .grepwidget.buttons.frame.search flash
      .grepwidget.buttons.frame.search invoke
   }
}

### Utility functions

proc setwordwrap {wrap} {
   if {$wrap} {	
     .results.text configure -wrap word
     grid forget .results.scrollx
   } else {
     .results.text configure -wrap none
     grid .results.scrollx -row 1 -column 0 -sticky nsew
   }
}

proc golookup {} {
   global g_searchword
   global g_senses

   if {[fixword $g_searchword] == ""} return
   .results.text configure \
      -state normal
   .results.text delete 1.0 end
   .results.text configure \
      -state disabled
   set g_senses {}
   .posmenubar.label configure \
      -text "Searches for $g_searchword:"
   if {[displayvalidsearchmenus $g_searchword]} {
      displayoverview $g_searchword
      history_add $g_searchword {} 0 0
   }
}

proc fixword {word} {
   regsub "^ *" $word "" word ; # remove leading spaces
   regsub " *$" $word "" word ; # remove trailing spaces
   regsub -all " " $word "_" word  ; # change other spaces to underscores
   return $word 
}

proc fixgrep {grepbuf word} {
    set greplines [split $grepbuf "\n"]
    foreach line $greplines {
	if { ([regexp -nocase "$word\$" $line]) && !([regexp " " $line]) } {
	    append retbuf $line
	    append retbuf "\n"
	}
    }
    return $retbuf
}

proc updatefonts {} {
   global fontsizelist
   global fontname
   global fontsize
   .results.text configure \
      -font "-adobe-$fontname-medium-r-*-*-*-[lindex \
         $fontsizelist $fontsize]-*-*-*-*-*-*"
   .results.text tag configure helpstyle \
      -font "-adobe-$fontname-medium-r-*-*-*-[lindex $fontsizelist \
         [expr $fontsize - 1]]-*-*-*-*-*-*" \
      -foreground Blue \
      -lmargin1 20
   .results.text tag configure overviewhighlight \
      -font "-adobe-$fontname-bold-r-*-*-*-[lindex $fontsizelist \
         $fontsize]-*-*-*-*-*-*" \
      -foreground Red
}

proc clearall {} {
   global g_searchword
   global g_senses
   set g_searchword {}
   set g_senses {}
   .results.text configure \
      -state normal
   .results.text delete 1.0 end
   .results.text configure \
      -state disabled
   foreach pos {noun verb adj adv} { pack forget .posmenubar.$pos }
   pack forget .wordframe.overview
   .statusbar.status configure \
      -text "Enter search word and press return."
   .posmenubar.label configure \
      -text ""
}

proc generaterange {low high} {
   for {set i $low} {$i <= $high} {incr i} {lappend res $i}
   return $res
}

proc getsenselist {senses} {
   regsub -all {[^0-9*]+} $senses " " senses
   regsub "^ *" $senses "" senses
   regsub "\ +$" $senses "" senses
   if {[regexp {\*} $senses] || ($senses == {})} { set senses 0 }
   return $senses
}

proc blackout {q} {
   set objectlist {
      .menubar.file
      .menubar.history
      .menubar.options
      .menubar.help
      .wordframe.entry
      .wordframe.overview
      .posmenubar.noun
      .posmenubar.verb
      .posmenubar.adj
      .posmenubar.adv
      .posmenubar.entry
   }
   if {$q} {
      bind . <KeyPress-Escape> {bell; abortsearch}
      bind .wordframe.entry <Return> {}
      bind .results.text <Shift-Button-1> {}
      bind .results.text <Button-2> {}
      bind . <Control-s> {}
      foreach object $objectlist {
         $object configure \
            -state disabled
      }
   } else {
      bind . <KeyPress-Escape> {}
      bind .wordframe.entry <Return> golookup
      bind .results.text <Shift-Button-1> {shiftclickhandler %x %y}
      bind .results.text <Button-2> {shiftclickhandler %x %y}
      bind . <Control-s> controlshandler
      foreach object $objectlist {
         $object configure \
            -state normal
      }
   }
}

proc shiftclickhandler {x y} { 
   global g_searchword
   set newsearchword [.results.text get "@$x,$y wordstart" "@$x,$y wordend"]
   if {$newsearchword != "\n"} {
      set g_searchword $newsearchword
      golookup
   }
}

proc controlshandler {} {
   global g_searchword
   if {[catch {selection get} newsearchword]} {
      bell
      return
   }
   if {$newsearchword != ""} {
      set g_searchword $newsearchword
      golookup
   }
}


### Visual Components

grid \
   [frame .menubar \
      -relief raised \
      -borderwidth 1] \
   -row 0 \
   -column 0 \
   -sticky ew
grid \
   [frame .wordframe \
      -relief raised \
      -borderwidth 1] \
   -row 1 \
   -column 0 \
   -sticky ew
grid \
   [frame .posmenubar \
      -relief raised \
      -borderwidth 1] \
   -row 2 \
   -column 0 \
   -sticky ew
grid \
   [frame .results \
      -relief raised \
      -borderwidth 1] \
   -row 3 \
   -column 0 \
   -sticky nsew \
   -ipadx 3 \
   -ipady 3
grid \
   [frame .statusbar \
      -relief raised \
      -borderwidth 1] \
   -row 4 \
   -column 0 \
   -sticky ew
grid rowconfigure . 0 -weight 0
grid rowconfigure . 1 -weight 0
grid rowconfigure . 2 -weight 0
grid rowconfigure . 3 -weight 1
grid rowconfigure . 4 -weight 0
grid columnconfigure . 0 -weight 1
pack \
   [menubutton .menubar.file \
      -text "File" \
      -menu .menubar.file.menu \
      -relief flat] \
   [menubutton .menubar.history \
      -text "History" \
      -menu .menubar.history.menu \
      -relief flat] \
   [menubutton .menubar.options \
      -text "Options" \
      -menu .menubar.options.menu \
      -relief flat] \
   [menubutton .menubar.help \
      -text "Help" \
      -menu .menubar.help.menu \
      -relief flat] \
   -side left
pack \
   [label .wordframe.label \
      -text "Search Word:"] \
   [entry .wordframe.entry \
      -textvariable g_searchword \
      -width 40 \
      -background White \
      -foreground Black] \
   -side left \
   -padx 2 \
   -pady 2
if {$tcl_platform(platform) != "macintosh"} {
   focus .wordframe.entry
}
button .wordframe.overview \
   -text "Redisplay Overview" \
   -padx 3 \
   -pady 1 \
   -highlightthickness 0 \
   -relief raised \
   -command {
      set g_senses {}
      displayoverview $g_searchword
      history_add $g_searchword {} 0 0
   }
pack \
   [label .posmenubar.label \
      -text ""] \
   -side left \
   -padx 2 \
   -pady 2
menubutton .posmenubar.noun \
   -text "Noun" \
   -menu .posmenubar.noun.menu \
   -relief raised
menubutton .posmenubar.verb \
   -text "Verb" \
   -menu .posmenubar.verb.menu \
   -relief raised
menubutton .posmenubar.adj \
   -text "Adjective" \
   -menu .posmenubar.adj.menu \
   -relief raised
menubutton .posmenubar.adv \
   -text "Adverb" \
   -menu .posmenubar.adv.menu \
   -relief raised
pack \
   [entry .posmenubar.entry \
      -textvariable g_senses \
      -width 10 \
      -background White \
      -foreground Black] \
   [label .posmenubar.senselabel \
      -text "Senses:"] \
   -side right \
   -padx 2 \
   -pady 2
grid \
   [text .results.text \
      -wrap word \
      -relief sunken \
      -borderwidth 2 \
      -state disabled \
      -yscrollcommand ".results.scrolly set" \
      -xscrollcommand ".results.scrollx set" \
      -background White \
      -foreground Black \
      -width 80 \
      -height 25 \
      -highlightthickness 0] \
   -row 0 \
   -column 0 \
   -sticky nsew
updatefonts
grid \
   [scrollbar .results.scrolly \
      -command ".results.text yview" \
      -width 12 \
      -relief sunken \
      -borderwidth 2 \
      -highlightthickness 0] \
   -row 0 \
   -column 1 \
   -sticky nsew
grid \
   [scrollbar .results.scrollx \
      -orient horizontal \
      -command ".results.text xview" \
      -width 12 \
      -relief sunken \
      -borderwidth 2 \
      -highlightthickness 0] \
   -row 1 \
   -column 0 \
   -sticky nsew
if {$wordwrap} {grid forget .results.scrollx}
grid rowconfigure .results 0 -weight 1
grid rowconfigure .results 1 -weight 0
grid columnconfigure .results 0 -weight 1
grid columnconfigure .results 1 -weight 0
pack \
   [label .statusbar.status \
      -text "Enter search word and press return."] \
   -side left \
   -padx 2

### Regular Menus

if {$labonly} {
menu .menubar.file.menu \
   -tearoff false
menu .menubar.history.menu \
   -tearoff false
menu .menubar.options.menu \
   -tearoff false
menu .menubar.options.menu.font \
   -tearoff false
menu .menubar.help.menu \
   -tearoff false
.menubar.file.menu add command \
   -label "Find keywords by substring" \
   -command "grepword substring"
.menubar.file.menu add command \
   -label "Find keywords by ending" \
   -command "grepword ending"
.menubar.file.menu add separator
.menubar.file.menu add command \
   -label "Save current display (specify path)" \
   -command "savetext main"
} else {
menu .menubar.file.menu \
   -tearoff false
menu .menubar.history.menu \
   -tearoff false
menu .menubar.options.menu \
   -tearoff false
menu .menubar.options.menu.font \
   -tearoff false
menu .menubar.help.menu \
   -tearoff false
.menubar.file.menu add command \
   -label "Find keywords by substring" \
   -command "grepword substring"
.menubar.file.menu add separator
.menubar.file.menu add command \
   -label "Save current display (specify path)" \
   -command "savetext main"
}

# Printing only works on Unix systems, so don't offer the option on
# other platforms.
if {$tcl_platform(platform) == "unix"} {
   .menubar.file.menu add command \
      -label "Print current display" \
      -command "printtext main"
}
.menubar.file.menu add command \
   -label "Clear current display" \
   -command clearall
.menubar.file.menu add separator
if {$labonly} {
   .menubar.file.menu add command \
      -label "Reopen database" \
      -command reopendb
   .menubar.file.menu add separator
}
.menubar.file.menu add command \
   -label "Exit" \
   -command { destroy . }
.menubar.options.menu add checkbutton \
   -label "Show help with each search" \
   -variable showcontextualhelp
.menubar.options.menu add checkbutton \
   -label "Show descriptive gloss" \
   -variable showglosses \
   -command { glosses $showglosses }
.menubar.options.menu add checkbutton \
   -label "Wrap lines" \
   -variable wordwrap \
   -command {.menubar.history.menu invoke 0} 
.menubar.options.menu add separator
.menubar.options.menu add command \
   -label "Set advanced search options..." \
   -command {
      toplevel .adviewopt
      wm title .adviewopt "Advanced search options"
      wm transient .adviewopt .
      scan [wm geometry .] "%dx%d+%d+%d" geom_h geom_w geom_x geom_y
      wm geometry .adviewopt +[expr $geom_x+50]+[expr $geom_y+50]
      wm resizable .adviewopt 0 0
      grab set .adviewopt
      pack \
         [frame .adviewopt.fileinfo \
            -relief raised \
            -borderwidth 1] \
         [frame .adviewopt.byteoffset \
            -relief raised \
            -borderwidth 1] \
         [frame .adviewopt.senseflag \
            -relief raised \
            -borderwidth 1] \
         [frame .adviewopt.bottom \
            -relief raised \
            -borderwidth 1] \
         -side top \
         -fill both \
         -expand true
      pack \
         [frame .adviewopt.fileinfo.f \
            -relief flat \
            -borderwidth 10] \
         -side top \
         -fill both \
         -expand true
      pack \
         [label .adviewopt.fileinfo.f.label \
            -text "Lexical file information"] \
         [radiobutton .adviewopt.fileinfo.f.opt0 \
            -text "Don't show" \
            -variable showfileinfo \
            -value 0] \
         [radiobutton .adviewopt.fileinfo.f.opt1 \
            -text "Show with searches" \
            -variable showfileinfo \
            -value 1] \
         [radiobutton .adviewopt.fileinfo.f.opt2 \
            -text "Show with searches and overview" \
            -variable showfileinfo \
            -value 2] \
         -side top \
         -anchor w
      pack \
         [frame .adviewopt.byteoffset.f \
            -relief flat \
            -borderwidth 10] \
         -side top \
         -fill both \
         -expand true
      pack \
         [label .adviewopt.byteoffset.f.label \
            -text "Synset location in database file"] \
         [radiobutton .adviewopt.byteoffset.f.opt0 \
            -text "Don't show" \
            -variable showbyteoffset \
            -value 0] \
         [radiobutton .adviewopt.byteoffset.f.opt1 \
            -text "Show with searches" \
            -variable showbyteoffset \
            -value 1] \
         [radiobutton .adviewopt.byteoffset.f.opt2 \
            -text "Show with searches and overview" \
            -variable showbyteoffset \
            -value 2] \
         -side top \
         -anchor w
      pack \
         [frame .adviewopt.senseflag.f \
            -relief flat \
            -borderwidth 10] \
         -side top \
         -fill both \
         -expand true
      pack \
         [label .adviewopt.senseflag.f.label \
            -text "Sense number"] \
         [radiobutton .adviewopt.senseflag.f.opt0 \
            -text "Don't show" \
            -variable showsenseflag \
            -value 0] \
         [radiobutton .adviewopt.senseflag.f.opt1 \
            -text "Show with searches" \
            -variable showsenseflag \
            -value 1] \
         [radiobutton .adviewopt.senseflag.f.opt2 \
            -text "Show with searches and overview" \
            -variable showsenseflag \
            -value 2] \
         -side top \
         -anchor w
      pack \
         [button .adviewopt.bottom.ok \
            -text "Ok" \
            -command "destroy .adviewopt"] \
         -side top \
         -pady 5
   }   
.menubar.options.menu add command \
   -label "Set maximum history length..." \
   -command setmaxhistorylength
.menubar.options.menu add command \
   -label "Set font..." \
   -command {
      toplevel .fontopt
      wm title .fontopt "Font"
      wm transient .fontopt .
      scan [wm geometry .] "%dx%d+%d+%d" geom_h geom_w geom_x geom_y
      wm geometry .fontopt +[expr $geom_x+50]+[expr $geom_y+50]
      wm resizable .fontopt 0 0
      grab set .fontopt
      pack \
         [frame .fontopt.top] \
         [frame .fontopt.bottom \
            -relief raised \
            -borderwidth 1] \
         -side top \
         -fill both \
         -expand true
      pack \
         [frame .fontopt.top.face \
            -relief raised \
            -borderwidth 1] \
         [frame .fontopt.top.size \
            -relief raised \
            -borderwidth 1] \
         -side left \
         -fill both \
         -expand true
      pack \
         [frame .fontopt.top.face.f \
            -relief flat \
            -borderwidth 10] \
         -side top \
         -fill both \
         -expand true
      pack \
         [label .fontopt.top.face.f.label \
            -text "Typeface"] \
         [radiobutton .fontopt.top.face.f.courier \
            -text "Courier" \
            -variable fontname \
            -value courier] \
         [radiobutton .fontopt.top.face.f.helvetica \
            -text "Helvetica" \
            -variable fontname \
            -value helvetica] \
         [radiobutton .fontopt.top.face.f.times \
            -text "Times" \
            -variable fontname \
            -value times] \
         -side top \
         -anchor w
      pack \
         [frame .fontopt.top.size.f \
            -relief flat \
            -borderwidth 10] \
         -side top \
         -fill both \
         -expand true
      pack \
         [label .fontopt.top.size.f.label \
            -text "Size"] \
         [radiobutton .fontopt.top.size.f.small \
            -text "Small" \
            -variable fontsize \
            -value 1] \
         [radiobutton .fontopt.top.size.f.medium \
            -text "Medium" \
            -variable fontsize \
            -value 2] \
         [radiobutton .fontopt.top.size.f.large \
            -text "Large" \
            -variable fontsize \
            -value 3] \
         -side top \
         -anchor w
      pack \
         [button .fontopt.bottom.ok \
            -text "Ok" \
            -command {
               updatefonts
               destroy .fontopt
	       .menubar.history.menu invoke 0
            }] \
         -side top \
         -pady 5
   }
.menubar.options.menu add separator
.menubar.options.menu add command \
    -label "Save current options as default" \
    -command {saveoptions}
.menubar.help.menu add command \
   -label "Help on using the WordNet browser" \
   -command {showhelpwidget helpwidget_xwn $resourcedir/wnb.man "WordNet Browser Help"}
.menubar.help.menu add command \
   -label "Help on WordNet terminology" \
   -command {showhelpwidget helpwidget_wngloss $resourcedir/wngloss.man "WordNet Glossary"}
.menubar.help.menu add command \
   -label "Display the WordNet license" \
   -command {showhelpwidget helpwidget_license $resourcedir/license.txt "WordNet License"}
.menubar.help.menu add separator
.menubar.help.menu add command \
   -label "About the WordNet browser" \
   -command showaboutbox

### Dynamic menus (for each part of speech)
### The numbers here are from wnconsts.h in the Wordnet library.
### Unfortunately, Tcl doesn't recognise #defined constants.
### Please be sure to keep them synchronised.

menu .posmenubar.noun.menu \
   -tearoff false
menu .posmenubar.verb.menu \
   -tearoff false
menu .posmenubar.adj.menu \
   -tearoff false
menu .posmenubar.adv.menu \
   -tearoff false
set posmenu(1) { 
   { "Synonyms, ordered by estimated frequency" 2 {} }
   { "Synonyms, grouped by similarity" 27 {} }
   { "Antonyms" 1 {} }
   { "Coordinate Terms" 26 {} }
   { "Hypernyms (this is a kind of...)" -2 {} }
   { "Hyponyms (...is a kind of this), brief" 3 {} }
   { "Hyponyms (...is a kind of this), full" -3 {} }
   { "Holonyms (this is a part of...), regular" 13 {} }
   { "Holonyms (this is a part of...), inherited" -29 {} }
   { "Meronyms (parts of this), regular" 12 {} }
   { "Meronyms (parts of this), inherited" -28 {} }
   { "Derivationally related forms" 20 {} }
   { "Attributes (...is a value of this)" 18 {} }
   { "Domain" 21 {} }
   { "Domain Terms" 22 {} }
   { "Familiarity" 24 {} }
}
set posmenu(2) {
   { "Synonyms, ordered by estimated frequency" 2 {} }
   { "Synonyms, grouped by similarity" 27 {} }
   { "Antonyms" 1 {} }
   { "Coordinate Terms" 26 {} }
   { "Hypernyms (this is one way to...)" -2 {} }
   { "Troponyms (particular ways to...), brief" 3 {} }
   { "Troponyms (particular ways to...), full" -3 {} }
   { "This entails doing..." 4 {} }
   { "This causes..." 14 {} }
   { "Derivationally related forms" 20 {} }
   { "Sentence frames" 25 {} }
   { "Domain" 21 {} }
   { "Domain Terms" 22 {} }
   { "Familiarity" 24 {} }
}
set posmenu(3) {
   { "Synonyms/Related Nouns" 5 {} }
   { "Antonyms" 1 {} }
   { "This is a value of..." 18 {} }
   { "Derivationally related forms" 20 {} }
   { "Domain" 21 {} }
   { "Domain Terms" 22 {} }
   { "Familiarity" 24 {} }
}
set posmenu(4) {
   { "Synonyms/Stem Adjectives" 23 {} }
   { "Antonyms" 1 {} }
   { "Derivationally related forms" 20 {} }
   { "Domain" 21 {} }
   { "Domain Terms" 22 {} }
   { "Familiarity" 24 {} }
}

### Bindings

bind .wordframe.entry <Return> golookup
bind .results.text <Shift-Button-1> {shiftclickhandler %x %y}
bind .results.text <Button-2> {shiftclickhandler %x %y}
bind . <Control-s> controlshandler
bind . <Control-g> "grepword substring"

set g_searchword [lindex $argv 0]
golookup

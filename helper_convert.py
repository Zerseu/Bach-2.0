import os
import re


class ConvertHelper:
    _validator = re.compile('^\|(Note|Chord|Rest|Bar).*')

    @staticmethod
    def _valid(s: str):
        return ConvertHelper._validator.match(s)

    @staticmethod
    def generate_input(ratio: float = 0.8):
        with open('data/training.txt', 'w') as training:
            with open('data/validation.txt', 'w') as validation:
                nwc = [f for f in os.listdir('data/input') if f.endswith('.nwctxt')]
                for file in nwc:
                    with open('data/input/' + file, 'r') as f:
                        content = f.readlines()
                        content = [string.replace('|', ' ').strip(' ') for string in content if
                                   ConvertHelper._valid(string)]
                        length = len(content)
                        split_point = int(length * ratio)
                        training.writelines(content[slice(0, split_point)])
                        validation.writelines(content[slice(split_point, length)])

    _header = """!NoteWorthyComposer(2.751)
    |Editor|ActiveStaff:1|CaretIndex:0|SelectIndex:0|CaretPos:0
    |SongInfo|Title:"Bach 2.0"|Author:"Johann Sebastian Bach"|Lyricist:"Alexandru-Ion Marinescu"
    |PgSetup|StaffSize:16|Zoom:4|TitlePage:Y|JustifyVertically:Y|PrintSystemSepMark:N|ExtendLastSystem:N|DurationPadding:Y|PageNumbers:0|StaffLabels:None|BarNumbers:None|StartingBar:1
    |Font|Style:StaffItalic|Typeface:"Times New Roman"|Size:10|Bold:Y|Italic:Y|CharSet:0
    |Font|Style:StaffBold|Typeface:"Times New Roman"|Size:8|Bold:Y|Italic:N|CharSet:0
    |Font|Style:StaffLyric|Typeface:"Times New Roman"|Size:7.2|Bold:N|Italic:N|CharSet:0
    |Font|Style:PageTitleText|Typeface:"Times New Roman"|Size:24|Bold:Y|Italic:N|CharSet:0
    |Font|Style:PageText|Typeface:"Times New Roman"|Size:12|Bold:N|Italic:N|CharSet:0
    |Font|Style:PageSmallText|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User1|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User2|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User3|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User4|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User5|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |Font|Style:User6|Typeface:"Times New Roman"|Size:8|Bold:N|Italic:N|CharSet:0
    |PgMargins|Left:1.27|Top:1.27|Right:1.27|Bottom:1.27|Mirror:N
    |AddStaff|Name:"Cello"|Group:"Standard"
    |StaffProperties|EndingBar:Section Close|Visible:Y|BoundaryTop:17|BoundaryBottom:12|Lines:5|Color:Default
    |StaffProperties|Muted:N|Volume:127|StereoPan:64|Device:0|Channel:1
    |StaffInstrument|Name:"String Ensemble 1"|Patch:48|Trans:7|DynVel:10,30,45,60,75,92,108,127
    |Clef|Type:Treble
    |Key|Signature:Bb|Tonic:F
    |Tempo|Tempo:100|Pos:10
    |TimeSig|Signature:6/8"""

    @staticmethod
    def generate_output():
        with open('data/output.nwctxt', 'r') as f:
            sentence = f.read()
        lines = sentence.split('<eos>')
        content = ''
        for line in lines:
            content += '|' + line.strip(' ') + '\n'
        content = content.replace(' ', '|')
        content = ConvertHelper._header + '\n' + content + '!NoteWorthyComposer-End'
        content = re.sub('^\|Note$', '', content, flags=re.MULTILINE)
        with open('data/output.nwctxt', 'w') as f:
            f.writelines(content)

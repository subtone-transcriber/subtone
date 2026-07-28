# heliper script to zip this project so I can easily upload it
(cd .. && zip -r subtone.zip subtone -x "stems*" -x "*__pycache__*" -x "*.pyc" -x "*__MACOSX*" -x "*/.*" -x "*egg-info*" -x "*testoutput*")

# Docker file for the nirs_sim_app plugin app

FROM fnndsc/ubuntu-python3:latest
MAINTAINER fnndsc "dev@babymri.org"

ENV APPROOT="/usr/src/nirs_sim_app"  VERSION="0.1"
COPY ["nirs_sim_app", "${APPROOT}"]
COPY ["requirements.txt", "${APPROOT}"]

WORKDIR $APPROOT

RUN pip install -r requirements.txt

CMD ["nirs_sim_app.py", "--json"]
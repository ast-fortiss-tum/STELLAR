FROM python:3.11-slim-bullseye

WORKDIR /src

COPY ./requirements.txt /src

RUN pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org

COPY . /src

ENTRYPOINT ["./run_navi_local.sh"]
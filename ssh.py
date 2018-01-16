from os import listdir
from os.path import join, isdir

exclude_dirs = ['.git', '.idea', 'cert']

def get_sh_files(ssh_dir = '/var/www/ssh'):
    sh = []

    for file in listdir(ssh_dir):
        if isdir(join(ssh_dir, file)) and file not in exclude_dirs:
            path = join(ssh_dir, file)
            sh.append({file: get_sh_files(path)})
        else:
            if file.endswith('.sh'):
                sh.append(join(ssh_dir, file))
    return sh

def parse_sh_file(path):
    with open(path) as file:
        sh_file = file.readlines()

    for line in sh_file:
        if 'USER' in line:
            eval(line)

    print(sh_file)
    print(USER)


print(get_sh_files())
parse_sh_file('/var/www/ssh/tds/offseason.sh')
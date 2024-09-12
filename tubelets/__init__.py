import .transforms as T

def build_transform(params: List[Dict]):
    transform_list = []
    for param in params:
        transform_list.append(getattr(T, **param))
        


    return Compose(transform_list)
